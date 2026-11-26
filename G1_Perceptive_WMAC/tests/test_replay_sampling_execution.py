"""Replay execution changes against the original prefix/draw/padding oracle."""
from __future__ import annotations

import copy
import unittest
from unittest.mock import patch

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from src.wmac.replay import ReplayBuffer,_Prefix
from src.wmac_perceptive.deduplicated_replay import DeduplicatedHikingReplay
from src.wmac_perceptive.hiking_amp import configure_replay
from test_deduplicated_replay import SPEC,Stream
from test_wmac_replay_fast import batch


class OriginalSampling:
    """Frozen pre-optimization integer prefix and draw, including pool order."""
    def _prefix(self,candidate_steps,split,kind):
        key=(candidate_steps,split,kind)
        if key in self._prefix_cache:return self._prefix_cache[key]
        if split not in self._SPLIT_GROUPS:raise ValueError('split must be train, validation or all')
        rows=self._row(torch.arange(candidate_steps,device=self.device))
        counts=self._group_counts[rows]
        tables=[]
        for table_kind in ('all','recovery','normal'):
            ids=tuple(group for group in self._SPLIT_GROUPS[split] if table_kind=='all'
                or (group%2==1)==(table_kind=='recovery'))
            groups=torch.tensor(ids,dtype=torch.long,device=self.device)
            cumulative=counts[:,groups].reshape(-1).cumsum(0)
            tables.append((table_kind,groups,cumulative))
        totals=torch.stack([cumulative[-1] if cumulative.numel() else self._group_counts.new_zeros(())
                            for _,_,cumulative in tables]).tolist()
        for (table_kind,groups,cumulative),total in zip(tables,totals):
            self._prefix_cache[(candidate_steps,split,table_kind)]=_Prefix(cumulative,groups,len(groups),int(total))
        self.index_cache_builds+=1
        return self._prefix_cache[key]

    def _draw(self,prefix,count):
        draws=torch.randint(prefix.total,(count,),device=self.device)
        slot=torch.searchsorted(prefix.cumulative,draws,right=True)
        age=slot//prefix.group_count
        group=prefix.groups[slot%prefix.group_count]
        before=torch.where(slot>0,prefix.cumulative[(slot-1).clamp_min(0)],0)
        row=self._row(age)
        offset=self._group_offsets[row,group]+draws-before
        return age,self._env_pool[row,offset].long()


class OriginalReplay(OriginalSampling,ReplayBuffer):
    def sample_windows(self,batch_size,length,recovery_fraction=.5,fields=None,split='train'):
        if length<1 or length>self.size:raise ValueError('Invalid original replay window')
        ages,env=self._sample_indices(batch_size,self.size-length+1,recovery_fraction,split)
        rows=self._row(ages[:,None]+torch.arange(length,device=self.device)[None])
        envs=env[:,None].expand_as(rows)
        valid=self._valid[rows,envs]
        boundary=self._boundaries[rows,envs]|self._cut_rows[rows,envs]
        mask=torch.cumprod(torch.cat((torch.ones(batch_size,1,device=self.device),
            (~boundary[:,:-1]).float()),1),1)*valid.float().cumprod(1)
        result={}
        for name in self._fields(fields):
            values=self.data[name][rows,envs]
            keep=valid.reshape(batch_size,length,*((1,)*(values.ndim-2)))
            result[name]=torch.where(keep,values,values[:,:1].expand_as(values))
        result.update(mask=mask,time_index=self._timestamps[rows],env_index=envs)
        return result


class OriginalDeduplicated(OriginalSampling,DeduplicatedHikingReplay):
    def _sample_windows(self,batch_size,length,recovery_fraction,fields,split,*,model=False):
        if length<1 or length>self.size:raise ValueError('Invalid original replay window')
        ages,env=self._sample_indices(batch_size,self.size-length+1,recovery_fraction,split)
        rows=self._row(ages[:,None]+torch.arange(length,device=self.device)[None])
        envs=env[:,None].expand_as(rows)
        valid=self._valid[rows,envs]
        boundary=self._boundaries[rows,envs]|self._cut_rows[rows,envs]
        mask=torch.cumprod(torch.cat((torch.ones(batch_size,1,device=self.device),
            (~boundary[:,:-1]).float()),1),1)*valid.float().cumprod(1)
        selected=self._fields(fields)
        if model:
            result=self._gather_fields(rows,envs,tuple(name for name in selected if name not in self._DEPTH_FIELDS))
            if 'depth_stack' in selected:result['depth_stack']=self._model_depth(rows[:,:1],envs[:,:1],False)
            if 'next_depth_stack' in selected:result['next_depth_stack']=self._model_depth(rows,envs,True)
            result={name:result[name] for name in selected}
        else:result=self._gather_fields(rows,envs,selected)
        for name,values in result.items():
            validity=valid[:,:1] if model and name=='depth_stack' else valid
            keep=validity.reshape(batch_size,validity.shape[1],*((1,)*(values.ndim-2)))
            result[name]=torch.where(keep,values,values[:,:1].expand_as(values))
        result.update(mask=mask,time_index=self._timestamps[rows],env_index=envs)
        return result


WM_FIELDS=('state','next_state','action','context','contact_cost','depth_stack','next_depth_stack','relative_height')


def deduplicated_pair(*,unusual=False,capacity=11):
    stream=Stream(unusual=unusual)
    stream.context=torch.cat((stream.context,stream.rand(stream.n,33)),-1)
    replays=[klass(capacity,stream.n,'cpu',.25,spec=SPEC,context_dim=133)
             for klass in (DeduplicatedHikingReplay,OriginalDeduplicated)]
    for replay in replays:
        replay.bootstrap_processed_history(stream.images)
        configure_replay(replay)
    return *replays,stream


class IndexCounts(TorchDispatchMode):
    def __init__(self,replay):
        super().__init__()
        self.replay=replay
        self.times=self.starts=0
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        if func==torch.ops.aten.index.Tensor:
            if args[0] is self.replay._timestamps:self.times+=1
            if args[0] is self.replay._episode_starts:self.starts+=1
        return func(*args,**(kwargs or {}))


class ReplaySamplingExecutionTest(unittest.TestCase):
    def same(self,actual,expected):
        self.assertEqual(tuple(actual),tuple(expected))
        for name,value in actual.items():
            other=expected[name]
            self.assertEqual((value.shape,value.dtype),(other.shape,other.dtype))
            if value.dtype==torch.float32:
                self.assertTrue(torch.equal(value.contiguous().view(torch.int32),other.contiguous().view(torch.int32)),name)
            else:torch.testing.assert_close(value,other,rtol=0,atol=0,equal_nan=True,msg=name)

    def compare_sequence(self,actual,original,method,*,split='train',fraction=0.,fields=None,length=3,count=10):
        def draw(replay):
            fn=getattr(replay,method)
            return (fn(73,recovery_fraction=fraction,fields=fields,split=split) if method=='sample_starts'
                    else fn(73,length,recovery_fraction=fraction,fields=fields,split=split))
        torch.manual_seed(7292)
        expected=[draw(original) for _ in range(count)]
        rng=torch.get_rng_state().clone()
        torch.manual_seed(7292)
        values=[draw(actual) for _ in range(count)]
        torch.testing.assert_close(torch.get_rng_state(),rng,rtol=0,atol=0)
        for value,truth in zip(values,expected):self.same(value,truth)

    def test_ten_wm_draws_all_splits_exact_rng_data_mask_and_codes(self):
        for unusual in (False,True):
            actual,original,stream=deduplicated_pair(unusual=unusual)
            for step in range(28):
                if step==17:
                    actual.mark_boundary();original.mark_boundary();stream.reset_all()
                item=stream.batch()
                actual.add(item);original.add(item)
            for split in ('train','validation','all'):
                for fraction in (0.,.5,1.):
                    self.compare_sequence(actual,original,'sample_model_windows',split=split,
                        fraction=fraction,fields=WM_FIELDS,length=7)
            self.assertEqual(actual._all_valid,not unusual)

    def test_general_replay_preserves_non_id_pool_order_boundary_masks_and_strata(self):
        actual,original=ReplayBuffer(9,10,validation_env_fraction=.2),OriginalReplay(9,10,validation_env_fraction=.2)
        for time in range(23):
            item=batch(time)
            item['boundary'][time%10]=True
            for replay in (actual,original):replay.add(item)
        for replay in (actual,original):
            for row in range(replay.capacity_steps):
                for group in range(4):
                    start=int(replay._group_offsets[row,group]);count=int(replay._group_counts[row,group])
                    replay._env_pool[row,start:start+count]=replay._env_pool[row,start:start+count].flip(0)
        for split in ('train','validation','all'):
            for fraction in (0.,.25,1.):
                self.compare_sequence(actual,original,'sample_windows',split=split,fraction=fraction,
                    fields=('next_state','action'),length=7,count=3)

    def test_uniform_model_sampling_avoids_readback_search_padding_and_repeated_metadata(self):
        actual,original,stream=deduplicated_pair()
        for _ in range(11):
            item=stream.batch();actual.add(item);original.add(item)
        torch.manual_seed(19)
        with IndexCounts(original) as old_indices,patch('torch.where',wraps=torch.where) as old_where:
            expected=original.sample_model_windows(73,7,fields=WM_FIELDS)
        torch.manual_seed(19)
        with (IndexCounts(actual) as new_indices,patch('torch.where',wraps=torch.where) as new_where,
                patch.object(torch.Tensor,'tolist',side_effect=AssertionError('Unexpected host readback')),
                patch('torch.searchsorted',side_effect=AssertionError('Unexpected binary search'))):
            value=actual.sample_model_windows(73,7,fields=WM_FIELDS)
        self.same(value,expected)
        self.assertEqual((old_indices.times,old_indices.starts),(3,2))
        self.assertEqual((new_indices.times,new_indices.starts),(1,1))
        self.assertEqual(old_where.call_count,9)
        self.assertEqual(new_where.call_count,0)

    def test_proof_is_conservative_after_invalid_overwrite_and_rebuilt_from_live_valid_on_load(self):
        actual,original=ReplayBuffer(4,10),OriginalReplay(4,10)
        for time in range(9):
            item=batch(time)
            if time==1:item['action'][0,0]=float('nan')
            for replay in (actual,original):replay.add(item)
        self.assertEqual(actual.invalid_transitions,1)
        self.assertFalse(actual._all_valid)
        self.assertTrue(bool(actual._valid.all()))
        self.compare_sequence(actual,original,'sample_windows',fraction=0.,length=3,count=3)
        saved=copy.deepcopy(actual.state_dict())
        self.assertNotIn('_all_valid',saved)
        restored=ReplayBuffer(4,10);restored.load_state_dict(saved)
        old=OriginalReplay(4,10);old.load_state_dict(saved)
        self.assertTrue(restored._all_valid)
        self.assertEqual(restored.invalid_transitions,1)
        self.compare_sequence(restored,old,'sample_windows',fraction=0.,length=3,count=3)

    def test_restore_does_not_trust_zero_cumulative_invalid_count(self):
        actual=ReplayBuffer(4,10)
        for time in range(4):
            item=batch(time)
            if time==2:item['action'][0,0]=float('nan')
            actual.add(item)
        saved=copy.deepcopy(actual.state_dict());saved['invalid_transitions']=0
        restored=ReplayBuffer(4,10);restored.load_state_dict(saved)
        original=OriginalReplay(4,10);original.load_state_dict(saved)
        self.assertFalse(restored._all_valid)
        self.assertEqual(restored.invalid_transitions,0)
        self.compare_sequence(restored,original,'sample_windows',fraction=0.,length=4,count=3)

    def test_field_subsets_and_start_sampling_remain_bit_exact(self):
        actual,original,stream=deduplicated_pair(unusual=True)
        for _ in range(19):
            item=stream.batch();actual.add(item);original.add(item)
        for fields in ((),('next_depth_stack','depth_stack'),('previous_action','next_context','actor_history')):
            for method in ('sample_starts','sample_windows','sample_model_windows'):
                self.compare_sequence(actual,original,method,fields=fields,count=2)

    def test_window_action_fields_keep_independent_contiguous_storage_after_padding_is_skipped(self):
        actual,original,stream=deduplicated_pair()
        for _ in range(11):
            item=stream.batch();actual.add(item);original.add(item)
        for method in ('sample_windows','sample_model_windows'):
            for length in (1,7):
                torch.manual_seed(71)
                expected=getattr(original,method)(17,length,0.,fields=('context','previous_action'))
                torch.manual_seed(71)
                result=getattr(actual,method)(17,length,0.,fields=('context','previous_action'))
                self.same(result,expected)
                self.assertEqual(result['previous_action'].stride(),expected['previous_action'].stride())
                actions=result['previous_action'].clone()
                result['context'][...,38:67].add_(100)
                torch.testing.assert_close(result['previous_action'],actions,rtol=0,atol=0)


if __name__=='__main__':unittest.main()
