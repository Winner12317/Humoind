"""Centred FP16 shared frames against full-stack replay, including terminal views."""
import copy
from dataclasses import replace
import unittest

import torch

from src.wmac_perceptive.centered_depth_replay import CenteredDepthReplay
from src.wmac_perceptive.hiking_amp import configure_replay
from src.wmac_perceptive.latent_replay import LatentPerceptiveReplay
from src.wmac_perceptive.perception import PerceptionSpec


SPEC = PerceptionSpec(height=2,width=3,ring_length=3,frames=2,stride=2,delay=0,
    crop_top=0,crop_bottom=0,crop_left=0,crop_right=0,blur_kernel=1,
    include_valid=False,depth_protocol="g1_rpl_v1")


def advance(history,new_frame):
    return torch.cat((history[:,1:],new_frame[:,None]),1)


def select_frames(history):
    return history.index_select(1,torch.tensor(SPEC.selected_indices))


def frame(generator):
    x = torch.rand(4,1,2,3,generator=generator)*1.4-.7
    x[0,0,0,0],x[1,0,0,0] = -.625,.625
    return x


def batch(step,current,following,*,bad=False,bad_image=False):
    state = torch.zeros(4,71)
    state[:,0] = step
    state[:,1] = torch.arange(4)
    state[:,2],state[:,3] = .8,1.
    next_state = state.clone();next_state[:,0] += .01
    if bad:next_state[1,0] = float('nan')
    now = select_frames(current).flatten(-4,-3)
    end = select_frames(following).flatten(-4,-3)
    if bad_image:
        end = end.clone();end[2,0,0,0] = float('nan')
    terminated,timeout,boundary = (torch.zeros(4,dtype=torch.bool) for _ in range(3))
    if step in (1,9,17):terminated[0] = True
    if step in (4,11,23):timeout[1] = True
    if step in (2,12):boundary[2] = True
    context = torch.zeros(4,35);context[2,34] = 3
    return dict(state=state,next_state=next_state,action=torch.zeros(4,29),context=context,
        relative_height=torch.tensor([.2,.8,.8,.8]),
        actor_history=torch.arange(4*SPEC.proprio_dim).reshape(4,SPEC.proprio_dim).float()*1e-5+step*.01,
        depth_stack=now,next_depth_stack=end,terminated=terminated,timeout=timeout,boundary=boundary)


def pair(steps=12,capacity=6,invalid=True,amp=False):
    g = torch.Generator().manual_seed(51)
    history = torch.stack([frame(g) for _ in range(SPEC.ring_length)],1)
    shared = CenteredDepthReplay(capacity,4,'cpu',.25,spec=SPEC)
    full = LatentPerceptiveReplay(capacity,4,'cpu',.25,spec=SPEC)
    if amp:
        configure_replay(shared);configure_replay(full)
    shared.bootstrap_processed_history(history)
    for step in range(steps):
        following = advance(history,frame(g))
        sample = batch(step,history,following,bad=invalid and step==3,bad_image=invalid and step==5)
        accepted = shared.add(sample)
        truth = full.add(sample)
        torch.testing.assert_close(accepted,truth,rtol=0,atol=0)
        done = sample['terminated']|sample['timeout']
        reset = frame(g)
        history = torch.where(done[:,None,None,None,None],reset[:,None].expand_as(following),following)
    return shared,full


class CenteredDepthReplayTests(unittest.TestCase):
    def assert_same_samples(self,shared,full,*,windows=False,split='train',fraction=0.):
        fields = ('state','next_state','context','actor_history','depth_stack','next_depth_stack','terminated','timeout')
        torch.manual_seed(14)
        actual = (shared.sample_windows(100,3,fraction,fields,split) if windows else
                  shared.sample_starts(100,fraction,fields,split))
        torch.manual_seed(14)
        expected = (full.sample_windows(100,3,fraction,fields,split) if windows else
                    full.sample_starts(100,fraction,fields,split))
        self.assertEqual(set(actual),set(expected))
        for name in actual:
            torch.testing.assert_close(actual[name],expected[name].to(actual[name].dtype),rtol=0,atol=0,equal_nan=True)

    def test_exact_fp16_images_proprioception_and_sampling(self):
        shared,full = pair(steps=6,capacity=8)
        self.assert_same_samples(shared,full)
        self.assert_same_samples(shared,full,windows=True)
        self.assertNotIn('depth_stack',shared.data)
        self.assertNotIn('next_depth_stack',shared.data)
        self.assertEqual(shared.current_frame_cache.dtype,torch.float16)
        self.assertEqual(shared.end_frame_cache.dtype,torch.float16)
        self.assertEqual(shared.data['actor_history'].dtype,torch.float32)
        sample = shared.sample_starts(100,fields=('depth_stack',))['depth_stack']
        self.assertLess(float(sample.min()),-.5)
        self.assertGreater(float(sample.max()),.5)

    def test_multiple_wraps_keep_history_and_distinct_terminal_view(self):
        shared,full = pair(steps=31,capacity=6,invalid=False)
        self.assertEqual((shared.ptr,shared.size,shared.total_steps),(1,6,31))
        self.assert_same_samples(shared,full)
        self.assert_same_samples(shared,full,windows=True)
        for age in range(shared.size):
            row = shared._row(torch.tensor(age)).expand(4)
            env = torch.arange(4)
            for name,successor in [('depth_stack',False),('next_depth_stack',True)]:
                torch.testing.assert_close(shared._depth(row,env,successor),full.data[name][row,env].float(),rtol=0,atol=0)

    def test_terminal_successor_is_not_replaced_by_next_reset_frame(self):
        g = torch.Generator().manual_seed(55)
        shared = CenteredDepthReplay(5,4,'cpu',spec=SPEC)
        first = frame(g)
        history = first[:,None].expand(-1,SPEC.ring_length,-1,-1,-1).clone()
        shared.bootstrap_processed_history(history)
        terminal_frame = torch.full_like(first,.6)
        terminal_history = advance(history,terminal_frame)
        sample = batch(0,history,terminal_history)
        sample['terminated'][0] = True
        shared.add(sample)
        reset_frame = torch.full_like(first,-.6)
        reset_history = reset_frame[:,None].expand_as(history).clone()
        next_history = advance(reset_history,torch.full_like(first,-.55))
        shared.add(batch(1,reset_history,next_history))
        row,env = torch.tensor([0]),torch.tensor([0])
        self.assertTrue(torch.equal(shared._depth(row,env,True)[0,-1],terminal_frame[0,0].half().float()))
        self.assertTrue(torch.equal(shared._depth(row+1,env)[0,-1],reset_frame[0,0].half().float()))

    def test_invalid_rows_windows_splits_and_strata_match_full_replay(self):
        shared,full = pair(steps=6,capacity=8)
        self.assertEqual(shared.invalid_transitions,full.invalid_transitions)
        for split in ('train','validation','all'):
            for fraction in (0.,.5,1.):
                self.assert_same_samples(shared,full,split=split,fraction=fraction)
                self.assert_same_samples(shared,full,windows=True,split=split,fraction=fraction)

    def test_bootstrap_and_schema_rejection_are_explicit(self):
        g = torch.Generator().manual_seed(6)
        history = torch.stack([frame(g) for _ in range(SPEC.ring_length)],1)
        sample = batch(0,history,advance(history,frame(g)))
        shared = CenteredDepthReplay(5,4,'cpu',spec=SPEC)
        with self.assertRaisesRegex(ValueError,'bootstrap'):
            shared.add(sample)
        shared.add(sample,processed_history=history)
        for name,value in [('action',torch.zeros(4,30)),('actor_history',torch.zeros(4,1))]:
            bad = {**sample,name:value}
            current = shared.current_frame_cache.clone();end = shared.end_frame_cache.clone()
            ptr,time = shared.ptr,shared.total_steps
            with self.assertRaises(ValueError):shared.add(bad)
            torch.testing.assert_close(shared.current_frame_cache,current,rtol=0,atol=0,equal_nan=True)
            torch.testing.assert_close(shared.end_frame_cache,end,rtol=0,atol=0,equal_nan=True)
            self.assertEqual((shared.ptr,shared.total_steps),(ptr,time))
        with self.assertRaises(ValueError):shared.bootstrap_processed_history(history)
        with self.assertRaisesRegex(ValueError,'delay0'):
            CenteredDepthReplay(5,4,'cpu',spec=replace(SPEC,delay=1,ring_length=4))

    def test_nonfinite_and_fp16_overflow_are_quarantined_without_clipping_valid_values(self):
        g = torch.Generator().manual_seed(7)
        first = frame(g)
        history = first[:,None].expand(-1,SPEC.ring_length,-1,-1,-1).clone()
        for bad in [float('nan'),float('inf'),70000.]:
            shared = CenteredDepthReplay(5,4,'cpu',spec=SPEC)
            broken = history.clone();broken[0,0,0,0,0] = bad
            with self.assertRaises(ValueError):shared.bootstrap_processed_history(broken)
            shared.bootstrap_reset(first)
            following = advance(history,frame(g))
            sample = batch(0,history,following)
            sample['next_depth_stack'][0,-1,0,0] = bad
            accepted = shared.add(sample)
            self.assertFalse(bool(accepted[0]))
            self.assertEqual(shared.invalid_transitions,1)
            selected = shared.sample_starts(100,fields=('depth_stack','next_depth_stack'))
            self.assertTrue(torch.isfinite(selected['depth_stack']).all())
            self.assertTrue(torch.isfinite(selected['next_depth_stack']).all())

    def test_restore_protocol_and_new_episode_separation(self):
        shared,full = pair(steps=12,capacity=6)
        payload = copy.deepcopy(shared.state_dict())
        restored = CenteredDepthReplay(6,4,'cpu',.25,spec=SPEC)
        restored.load_state_dict(payload)
        self.assert_same_samples(restored,full)
        self.assert_same_samples(restored,full,windows=True)
        for wrong in [{**payload,'perceptive_format':'frame_uint8_v1'},full.state_dict()]:
            with self.assertRaisesRegex(ValueError,'protocol'):restored.load_state_dict(wrong)
        g = torch.Generator().manual_seed(87)
        new = frame(g)
        history = new[:,None].expand(-1,SPEC.ring_length,-1,-1,-1).clone()
        sample = batch(12,history,advance(history,frame(g)))
        restored.add(sample)
        row = torch.full((4,),payload['ptr'],dtype=torch.long)
        torch.testing.assert_close(restored._depth(row,torch.arange(4)),sample['depth_stack'].half().float(),rtol=0,atol=0)

    def test_hiking_amp_histories_restore_across_wraps(self):
        shared,full = pair(steps=31,capacity=6,invalid=False,amp=True)
        self.assert_same_samples(shared,full)
        restored = CenteredDepthReplay(6,4,'cpu',.25,spec=SPEC)
        restored.load_state_dict(copy.deepcopy(shared.state_dict()))
        self.assertTrue(restored._hiking_amp_history)
        self.assert_same_samples(restored,full)
        self.assertTrue(bool(restored._amp_pending_reset.all()))

    def test_budget_reports_both_fp16_frame_caches(self):
        expected = (2*1000+36)*4096*27*48*2
        self.assertEqual(CenteredDepthReplay.estimated_frame_cache_bytes(1000,4096),expected)
        shared = CenteredDepthReplay(5,4,'cpu',spec=SPEC)
        actual = sum(x.numel()*x.element_size() for x in [shared.current_frame_cache,shared.end_frame_cache])
        self.assertEqual(actual,CenteredDepthReplay.estimated_frame_cache_bytes(5,4,SPEC))


if __name__ == '__main__':
    unittest.main()
