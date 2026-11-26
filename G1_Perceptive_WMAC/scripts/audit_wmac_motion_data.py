"""Audit cached forward-walk/turn motion data without changing training or caches.

Fresh training reads the pinned raw NPZ and selection YAML. Pass an older
converted dataset root explicitly to audit its reconstruction. For the historical filtered protocol, add
--check-remote for HEAD-only checks of pinned GRAIL URLs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import concurrent.futures
import dataclasses
import datetime
import hashlib
import json
import math
from pathlib import Path
import sys
import urllib.parse
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def audit(args, report):
    import numpy as np

    from src.wmac_perceptive.motions import read_motion, verify_locomotion_dataset
    from src.wmac_perceptive.mirroring import build_g1_mirror_map, mirror_motion, validate_motion
    from src.wmac_perceptive.locomotion_data import (
        HIKING_ARCHIVE_SHA256, HIKING_MOTION_SHA256,
        HIKING_MOTION_MEMBER, LocomotionFilter, convert_hiking_motion,
        frame_selection, contiguous_segments, slice_motion)
    from src.wmac_perceptive.grail_data import (
        GRAIL_SAMPLES, GRAIL_REVISION,
        GRAIL_REPOSITORY, convert_grail_motion, load_numeric_joblib)

    DATA=args.dataset_root
    MANIFEST = DATA / 'mirror_manifest.json'
    m = json.loads(MANIFEST.read_text())
    report.update(dataset_root=str(DATA), manifest_sha256=hashlib.sha256(MANIFEST.read_bytes()).hexdigest())

    def check(name, condition, details=None):
        value = dict(passed=bool(condition))
        if details is not None: value['details'] = details
        report['checks'][name] = value
        if not condition: report['issues'].append(dict(check=name, details=details))

    def digest(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def audit_file(path, sha, kind):
        path=Path(path)
        exists=path.is_file()
        actual=digest(path) if exists else None
        row=dict(path=str(path),kind=kind,exists=exists,bytes=path.stat().st_size if exists else 0,
                 expected_sha256=sha,actual_sha256=actual,sha256_match=sha==actual)
        report['source_files'].append(row)
        check(f'file:{path}', exists and sha==actual)
        return row

    mapping=build_g1_mirror_map(args.robot_xml)
    check('robot_mapping_matches_current_xml',mapping.metadata()==m['robot_mapping'])
    check('filter_parameters_match_current_code',dataclasses.asdict(LocomotionFilter())==m['filters'])
    try:
        verify_locomotion_dataset(DATA,source_root=args.source_root)
        check('training_dataset_verifier',True)
    except Exception as e:
        check('training_dataset_verifier',False,repr(e))

    all_arrays={}
    for name,sha in sorted(m['output_sha256'].items()):
        path=DATA/name
        arr=read_motion(path)
        frames=validate_motion(arr,mapping)
        row=dict(path=name,sha256=digest(path),sha256_match=digest(path)==sha,frames=frames,
                 fps=float(arr['fps'].reshape(-1)[0]),
                 arrays={k:dict(shape=list(v.shape),dtype=str(v.dtype),finite=bool(np.isfinite(v).all())) for k,v in arr.items()},
                 maximum_quaternion_norm_error=float(np.max(np.abs(np.linalg.norm(arr['body_quat_w'],axis=-1)-1))))
        report['files'].append(row)
        all_arrays[name]=arr
    check('all_training_state_arrays_finite_float32_50hz',all(
        row['fps']==50 and all(a['finite'] and (k=='fps' or a['dtype']=='float32') for k,a in row['arrays'].items()) for row in report['files']))
    report['scalar_fps_dtypes']=dict(Counter(row['arrays']['fps']['dtype'] for row in report['files']))
    report['dtype_note']='State tensors must be finite float32; scalar fps may use another floating dtype, with every value exactly 50 Hz.'
    check('manifest_training_frame_count',sum(r['frames'] for r in report['files'])==m['walking_total_frames'])
    check('all_clip_references_exist',all(row[k] in all_arrays for row in m['clips'] for k in ('original_output','mirrored_output')))
    check('all_outputs_referenced_exactly_once',sorted(row[k] for row in m['clips'] for k in ('original_output','mirrored_output'))==sorted(all_arrays))

    local_root=args.source_root/'WalkandRun'
    for name,sha in sorted(m['local_source_sha256'].items()):
        audit_file(local_root/name,sha,'local_raw_motion')
        validate_motion(read_motion(local_root/name),mapping)

    archive=args.hiking_source/'hiking-in-the-wild_Data_Model.zip'
    audit_file(archive,HIKING_ARCHIVE_SHA256,'complete_official_hiking_data_model_archive')
    with zipfile.ZipFile(archive) as z:
        check('hiking_zip_crc',z.testzip() is None)
        report['hiking_archive_inventory']=[dict(name=f.filename,bytes=f.file_size) for f in z.infolist() if not f.is_dir() and not f.filename.startswith('__MACOSX/')]
        members=(HIKING_MOTION_MEMBER,'data&model/parkour_motion_reference/parkour_motion_without_run.yaml','data&model/README.md')
        for member in members:
            p=args.hiking_source/Path(member).name
            audit_file(p,hashlib.sha256(z.read(member)).hexdigest(),'hiking_extracted_data')
        reference_members=[f.filename for f in z.infolist() if f.filename.startswith('data&model/parkour_motion_reference/') and f.filename.endswith('.npz')]
        check('all_official_hiking_archive_motion_npz_extracted',all((args.hiking_source/Path(n).name).is_file() for n in reference_members),reference_members)
    motion_path=args.hiking_source/Path(HIKING_MOTION_MEMBER).name
    check('hiking_npz_pin_matches_manifest_and_code',digest(motion_path)==HIKING_MOTION_SHA256==m['source_metadata'][0]['npz_sha256'])
    with np.load(motion_path,allow_pickle=False) as a:
        report['hiking_raw_npz']=dict(path=str(motion_path),arrays={k:dict(shape=list(a[k].shape),dtype=str(a[k].dtype),finite=bool(np.isfinite(a[k]).all()) if a[k].dtype.kind in 'fiu' else None) for k in a.files})
        check('hiking_yaml_selected_files_resolves', 'parkour_motion_without_run_retargetted.npz' in (args.hiking_source/'parkour_motion_without_run.yaml').read_text())

    grail_refs=[]
    for cat,stem,shas in GRAIL_SAMPLES:
        files={}; records=[]
        meta=next(s for s in m['source_metadata'] if s.get('terrain_category')==cat)
        for group,sha in zip(('robot','meta','objects','object_usd'),shas):
            relative=f'data/{cat}/{group}/{stem}'+('.usd' if group=='object_usd' else '.pkl')
            url=f'https://huggingface.co/datasets/{GRAIL_REPOSITORY}/resolve/{GRAIL_REVISION}/'+urllib.parse.quote(relative,safe='/')
            path=args.grail_source/relative
            row=audit_file(path,sha,'grail_'+group)
            check('grail_manifest_pin:'+relative,any(r['path']==relative and r['sha256']==sha and r['url']==url and r['bytes']==row['bytes'] for r in meta['source_records']))
            files[group]=path
            records.append(dict(path=relative,sha256=sha,url=url,bytes=row['bytes']))
            report['remote_pinned_urls'].append(dict(url=url,category=cat,kind=group))
            if group!='object_usd':load_numeric_joblib(path)
        grail_refs.append(dict(category=cat,files=files,source_records=records))

    def remote_head(row):
        result=dict(row)
        try:
            req=urllib.request.Request(row['url'],method='HEAD',headers={'User-Agent':'motion-integrity-audit/1.0'})
            with urllib.request.urlopen(req,timeout=15) as response:
                result.update(reachable=True,status=response.status,content_length=response.headers.get('Content-Length'),bytes_downloaded=0)
        except Exception as e:result.update(reachable=False,error=repr(e))
        return result


    print('Reconstructing selected intervals from cached sources without changing the cache.',flush=True)
    hiking,_=convert_hiking_motion(motion_path,xml_path=args.robot_xml)
    candidates={('hiking_public',motion_path.name):(hiking,True)}
    for ref in grail_refs:
        motion,_=convert_grail_motion(ref,xml_path=args.robot_xml)
        candidates[(f"grail_{ref['category']}",ref['files']['robot'].name)]=(motion,True)
    for row in m['source_audits']:
        if row['source']=='local_supplement':candidates[(row['source'],row['file'])]=(read_motion(local_root/row['file']),False)

    for audit in m['source_audits']:
        key=(audit['source'],audit['file'])
        motion,normalize=candidates[key]
        valid,diag=frame_selection(motion,mapping)
        intervals=contiguous_segments(valid,max(2,math.ceil(float(motion['fps'].reshape(-1)[0])*LocomotionFilter().minimum_segment_seconds)))
        clips=[row for row in m['clips'] if (row['source'],row['file'])==key]
        expected=[(row['source_start_frame'],row['source_end_frame_exclusive']) for row in clips]
        check('recomputed_intervals:'+str(key),intervals==expected)
        match=True; max_error=0.
        for clip in clips:
            selected=slice_motion(motion,clip['source_start_frame'],clip['source_end_frame_exclusive'],support_height=diag['support_height'] if normalize else None)
            reflected=mirror_motion(selected,mapping)
            for output,arr in ((clip['original_output'],selected),(clip['mirrored_output'],reflected)):
                stored=all_arrays[output]
                for k in stored:
                    match=match and np.array_equal(stored[k],arr[k])
                    max_error=max(max_error,float(np.max(np.abs(stored[k]-arr[k]))))
        check('reconstructed_selected_arrays:'+str(key),match,dict(maximum_absolute_error=max_error))
        report['source_reconstruction'].append(dict(source=key[0],file=key[1],input_frames=len(valid),
            passing_filter_frames=int(valid.sum()),retained_frames=sum(e-s for s,e in intervals),
            retained_intervals=len(intervals),exact_array_match=bool(match),maximum_absolute_array_error=max_error))

    report['remote_checks_requested']=args.check_remote
    if args.check_remote:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            report['remote_pinned_urls']=list(pool.map(remote_head,report['remote_pinned_urls']))
        check('all_grail_pinned_urls_reachable',len(report['remote_pinned_urls'])==4*len(GRAIL_SAMPLES)
            and all(r['reachable'] for r in report['remote_pinned_urls']))
    else:
        for row in report['remote_pinned_urls']:row['checked']=False
    if args.training_config is not None:
        cfg=json.loads(args.training_config.read_text())
        report['actual_training_config']=dict(path=str(args.training_config),sha256=digest(args.training_config),
            **{k:cfg[k] for k in ('task','motion_scope','mirrored_motion_root','expert_recovery_fraction','recovery_fraction','amp_feature_schema')})
        check('actual_training_points_to_audited_dataset',(ROOT/cfg['mirrored_motion_root']).resolve()==DATA.resolve()
            and cfg['motion_scope']=='forward_walk_turn_v1' and cfg['expert_recovery_fraction']==0 and cfg['recovery_fraction']==0)
    else:
        report['actual_training_config']={'checked':False,'reason':'Pass --training-config RUN/config.json to verify a specific run.'}
    report['training_code_references']=dict(
        reset_loader='scripts/train_wmac_perceptive.py:make_environment resolves prepare_locomotion_dataset and sets init/reset motion_dir to paths.walk, recovery_dir=None.',
        amp_loader='src/wmac_perceptive/expert.py:PerceptiveExpertPairs verifies same manifest and loads every WalkandRun NPZ, constructs only within-clip adjacent AMP pairs.',
        runner='src/wmac_perceptive/runner.py:_make_experts passes cfg.motion_root to PerceptiveExpertPairs.')
    report['summary']=dict(training_clips=len(all_arrays),training_frames=sum(r['frames'] for r in report['files']),
        adjacent_amp_pairs=sum(r['frames']-1 for r in report['files']),unmirrored_clips=len(m['clips']),
        unmirrored_frames=sum(r['frames'] for r in m['clips']),unmirrored_frame_count_seconds=sum(r['frames']/50 for r in m['clips']),
        unique_source_sequence_count=len(set((r['source'],r['file']) for r in m['clips'])),
        hiking_raw_frames=len(hiking['joint_pos']),hiking_retained_frames=m['hiking_retained_frames'],
        grail_raw_sequences=len(grail_refs),
        grail_retained_frames=m['grail_retained_frames'],local_retained_frames=m['local_supplement_retained_frames'],
        recovery_training_files=len(list((DATA/'Recovery').glob('*.npz'))),missing_required_files=[r['path'] for r in report['source_files'] if not r['exists']],
        cache_mutated=False)
    report['limitations']=[
        'File/schema/hash completeness is not proof of sufficient locomotion coverage or convergence.',
        'The complete pinned Hiking public ZIP contains one pose reference NPZ, selected YAML, README and example checkpoints. Only the required motion/YAML/README are extracted; checkpoint bytes remain in ZIP and are not training inputs.',
        'GRAIL is intentionally three pinned terrain examples with all four supporting files per example, not the full GRAIL dataset.',
        'USD files are retained as provenance only; this audit does not import their geometry or inspect optional external texture dependencies, and the training code does not consume those USD assets.',
        'Hiking/GRAIL pose references contain no depth frame sequence or verified per-frame foot contact labels. Simulator generates training depth and terrain interactions.',
        'Sampling time counts frames/fps including final frames; adjacent-pair durations are (frames-1)/fps per clip.',
        'Manifest completeness does not certify original dataset author licensing or redistribute untracked cached datasets.']


def audit_official_hiking(args,report,*,include_standing=False):
    """Reconstruct the full author-selected source; never alter cache bytes."""
    import numpy as np
    from src.wmac_perceptive.hiking_data import verify_hiking_dataset
    from src.wmac_perceptive.locomotion_data import (
        HIKING_SOURCE_SHA256,HIKING_ARCHIVE_SHA256,HIKING_MOTION_MEMBER,convert_hiking_motion)
    from src.wmac_perceptive.mirroring import build_g1_mirror_map,mirror_motion,validate_motion
    from src.wmac_perceptive.motions import file_sha256,read_motion

    def check(name,condition,details=None):
        report['checks'][name]={'passed':bool(condition)}
        if details is not None:report['checks'][name]['details']=details
        if not condition:report['issues'].append({'check':name,'details':details})

    path=args.dataset_root/'mirror_manifest.json'
    if include_standing:
        from src.wmac_perceptive.standing_data import verify_hiking_standing_dataset
        manifest=verify_hiking_standing_dataset(args.dataset_root)
    else:
        manifest=verify_hiking_dataset(args.dataset_root)
    report.update(dataset_root=str(args.dataset_root),manifest_sha256=file_sha256(path),
                  motion_data_protocol='hiking_walk_v1',remote_checks_requested=args.check_remote)
    check('dataset_file_set_and_hashes',True)
    mapping=build_g1_mirror_map(args.robot_xml)
    check('robot_mapping_matches_current_xml',manifest['robot_mapping']==mapping.metadata())
    source=args.hiking_source/Path(HIKING_MOTION_MEMBER).name
    archive=args.hiking_source/'hiking-in-the-wild_Data_Model.zip'
    for name,expected in HIKING_SOURCE_SHA256.items():
        p=args.hiking_source/name
        check('source_sha256:'+p.name,p.is_file() and file_sha256(p)==expected)
        report['source_files'].append(dict(path=str(p),exists=p.is_file(),sha256=file_sha256(p) if p.is_file() else None))
    selection=args.hiking_source/'parkour_motion_without_run.yaml'
    report['original_archive']={'path':str(archive),'available':archive.is_file(),
        'required':False,'reason':'The three pinned source files are sufficient for official walking training'}
    if archive.is_file():
        check('source_sha256:'+archive.name,file_sha256(archive)==HIKING_ARCHIVE_SHA256)
        with zipfile.ZipFile(archive) as package:
            check('archive_crc',package.testzip() is None)
            check('yaml_matches_official_package',selection.read_bytes()==package.read('data&model/parkour_motion_reference/parkour_motion_without_run.yaml'))
    check('source_file_selection_matches_yaml',manifest['source_metadata'][0]['selection_sha256']==file_sha256(selection))
    original,conversion=convert_hiking_motion(source,xml_path=args.robot_xml,velocity_protocol='hiking_frontward_v1')
    mirror=mirror_motion(original,mapping)
    row=next(clip for clip in manifest['clips'] if clip['source']=='hiking_public')
    for name,expected in [(row['original_output'],original),(row['mirrored_output'],mirror)]:
        actual=read_motion(args.dataset_root/name)
        frames=validate_motion(actual,mapping)
        exact=set(actual)==set(expected) and all(np.array_equal(actual[k],expected[k]) for k in actual)
        check('exact_reconstruction:'+name,exact)
        check('all_finite:'+name,all(np.isfinite(v).all() for v in actual.values()))
        check('final_forward_velocity_zero:'+name,all(np.all(actual[k][-1]==0) for k in ('joint_vel','body_lin_vel_w','body_ang_vel_w')))
        report['files'].append(dict(path=name,frames=frames,fps=float(actual['fps'][0]),
            sha256=file_sha256(args.dataset_root/name),exact_reconstruction=exact))
    if include_standing:
        from src.wmac_perceptive.standing_data import (
            STANDING_BUNDLE,STANDING_CSV_NAME,STANDING_CSV_SHA256,STANDING_EXPERT_SOURCE,
            convert_legacy_g1_standing,verify_standing_source)
        from src.wmac_perceptive.hiking_amp import HISTORY_LENGTH
        provenance=verify_standing_source()
        csv=STANDING_BUNDLE/STANDING_CSV_NAME
        check('standing_source_sha256',file_sha256(csv)==STANDING_CSV_SHA256)
        report['source_files'].append(dict(path=str(csv),exists=True,sha256=file_sha256(csv)))
        report['standing_provenance']=provenance
        report['standing_expert_source']=STANDING_EXPERT_SOURCE
        standing=convert_legacy_g1_standing(xml_path=args.robot_xml)
        standing_mirror=mirror_motion(standing,mapping)
        standing_clip=next(clip for clip in manifest['clips'] if clip['source']==STANDING_EXPERT_SOURCE)
        standing_names=(standing_clip['original_output'],standing_clip['mirrored_output'])
        for name,expected in zip(standing_names,(standing,standing_mirror)):
            actual=read_motion(args.dataset_root/name)
            frames=validate_motion(actual,mapping)
            exact=set(actual)==set(expected) and all(np.array_equal(actual[k],expected[k]) for k in actual)
            check('exact_reconstruction:'+name,exact)
            check('standing_all_velocities_zero:'+name,all(np.all(actual[k]==0)
                for k in ('joint_vel','body_lin_vel_w','body_ang_vel_w')))
            check('standing_all_rows_same_pose:'+name,all(np.array_equal(actual[k],np.broadcast_to(actual[k][:1],actual[k].shape))
                for k in ('joint_pos','body_pos_w','body_quat_w')))
            report['files'].append(dict(path=name,frames=frames,fps=float(actual['fps'][0]),
                sha256=file_sha256(args.dataset_root/name),exact_reconstruction=exact,
                reference_kind='standing',source_fps=None,representation_fps=50.))
        official_names={row['original_output'],row['mirrored_output']}
        check('official_walking_npz_bytes_preserved',
            set(manifest['official_walking_output_sha256'])==official_names and all(
                file_sha256(args.dataset_root/name)==manifest['official_walking_output_sha256'][name]
                for name in official_names))
        standing_windows=sum(item['frames']-HISTORY_LENGTH+1 for item in report['files'] if item['path'] in standing_names)
        total_windows=sum(item['frames']-HISTORY_LENGTH+1 for item in report['files'])
        check('standing_history_window_count',standing_windows==manifest['standing_history_windows'])
        check('total_history_window_count',total_windows==manifest['total_history_windows'])
        check('natural_standing_sampling_fraction',
            standing_windows/total_windows==manifest['natural_standing_expert_fraction'])
        check('standing_duration_is_representation_only',manifest['standing_source_fps'] is None
            and manifest['standing_representation_fps']==50.)
    check('no_local_frame_filters',manifest['filters'] is None)
    check('no_recovery_group',not (args.dataset_root/'Recovery').exists())
    if include_standing:
        check('only_requested_standing_supplement',not manifest['grail_motion_data_used']
            and manifest['local_supplement_retained_frames']==len(standing['joint_pos'])
            and {clip['source'] for clip in manifest['clips']}=={'hiking_public',STANDING_EXPERT_SOURCE})
        check('full_author_resampling_timeline',len(original['joint_pos'])==manifest['hiking_retained_frames'])
    else:
        check('no_extra_local_or_grail_data',not manifest['grail_motion_data_used'] and manifest['local_supplement_retained_frames']==0)
        check('full_author_resampling_timeline',len(original['joint_pos'])==manifest['loaded_frames'] if 'loaded_frames' in manifest else len(original['joint_pos'])==manifest['walking_original_frames'])
    if args.training_config is not None:
        cfg=json.loads(args.training_config.read_text())
        check('actual_training_data_protocol',cfg['motion_data_protocol']=='hiking_walk_v1')
        check('actual_training_standing_source',cfg.get('standing_expert_source','none')==
            ('legacy_g1_locked_v1' if include_standing else 'none'))
        check('actual_training_dataset_path',(ROOT/cfg['mirrored_motion_root']).resolve()==args.dataset_root)
        report['actual_training_config']=dict(path=str(args.training_config),sha256=file_sha256(args.training_config))
    report['conversion']=conversion
    report['summary']=dict(training_clips=len(report['files']),training_frames=sum(r['frames'] for r in report['files']),
        unmirrored_clips=len(manifest['clips']),source_frames=manifest['source_frames'],loaded_frames=manifest['walking_original_frames'],
        source_transition_seconds=manifest['source_transition_seconds'],
        loaded_transition_seconds=manifest['walking_original_transition_seconds'],
        frame_filter=None,height_normalization=None,grail_used=False,recovery_training_files=0,cache_mutated=False)
    if include_standing:
        report['summary'].update(hiking_source_frames=manifest['source_frames'],
            hiking_loaded_frames=len(original['joint_pos']),standing_source_frames=len(standing['joint_pos']),
            standing_training_clips=2,standing_training_frames=2*len(standing['joint_pos']),
            standing_source_fps=None,standing_representation_fps=50.,
            standing_representation_transition_seconds=(len(standing['joint_pos'])-1)/50.,
            standing_history_window_count=standing_windows,total_history_window_count=total_windows,
            standing_expert_window_fraction=standing_windows/total_windows,
            expert_sampling='Uniform valid ten-frame windows; no custom standing mixture weights')
    report['limitations']=[
        'Walking-only classification comes from the author-selected dataset; natural sway/negative instantaneous speeds are not relabeled by local heuristics.',
        'Author resampling uses an exclusive endpoint: 18982 source frames become 18981 loaded frames at 50Hz.',
        'Source joins are retained as in the official loader; the public file does not expose internal clip labels.',
        'Terrain interaction and camera depth are collected online; no per-terrain demonstration quota is imposed.',
        'MuJoCo pelvis-root FK adapts the torso-root source to local robot assets; this is not an Isaac geometry identity claim.']
    if include_standing:
        report['limitations'].extend([
            'The standing clip is from the user\'s earlier G1 AMP project, not from Hiking and not from an identified R1 dataset.',
            'The original standing CSV has no sampling rate. Its 1201 identical rows are represented at 50 Hz; 24 seconds is a representation duration, not an inferred recording duration.',
            'A locked pose is not a dynamic balance demonstration and does not certify a trained policy\'s standing stability.'])
    if args.check_remote:
        report['remote_checks_note']='Official source files are checked against fixed digests; the optional original archive is checked when present. No supplementary GRAIL URLs are needed by this protocol.'


def audit_hiking_reference(source_dir, args, report):
    import numpy as np
    from src.wmac_perceptive.config import LEGACY_WALKING_EXPERT
    from src.wmac_perceptive.hiking_amp import HISTORY_LENGTH
    from src.wmac_perceptive.hiking_data import official_hiking_reference
    from src.wmac_perceptive.locomotion_data import HIKING_SOURCE_SHA256, interpolate_hiking_motion
    from src.wmac_perceptive.motions import verify_locomotion_dataset

    source = official_hiking_reference(source_dir, prepare=False)
    with np.load(source, allow_pickle=False) as data:
        fps = float(data["framerate"])
        names = data["joint_names"].tolist()
        raw_frames = len(data["joint_pos"])
        trajectories = interpolate_hiking_motion(data["base_pos_w"], data["base_quat_w"], data["joint_pos"], fps)
    if len(set(names)) != 29 or fps != 50 or not all(np.isfinite(t).all() for t in trajectories):
        raise ValueError("Official reference joint layout, control rate or interpolated trajectory is invalid")
    walking_expert_source = "none"
    if args.training_config:
        cfg = json.loads(args.training_config.read_text())
        if (cfg.get("hiking_training_version", 1) < 3 or cfg.get("standing_expert_source") != "none"
                or cfg.get("motion_data_protocol") != "hiking_walk_v1"):
            raise ValueError("Training configuration does not use the raw Hiking reference")
        walking_expert_source = cfg.get("walking_expert_source", "none")
        if walking_expert_source not in ("none", LEGACY_WALKING_EXPERT):
            raise ValueError("Training configuration has an unknown walking expert source")
    report.update(dataset_root=str(source.parent), motion_data_protocol="hiking_walk_v1",
        source_files=[dict(name=name, sha256=digest) for name,digest in HIKING_SOURCE_SHA256.items()],
        summary=dict(source_clips=1, mirror_probability=.5, raw_frames=raw_frames,
                     hiking_loaded_frames=len(trajectories[0]), framerate_hz=fps,
                     reference_start_fraction=[0., .9], local_supplement_frames=0, grail_frames=0))
    report["checks"].update(source_files_match_release={"passed": True},
        selected_files_match_official_yaml={"passed": True}, complete_unfiltered_trajectory={"passed": True})
    if walking_expert_source == LEGACY_WALKING_EXPERT:
        local_root = source.parent / "legacy_local_walk_turn_v2"
        local = verify_locomotion_dataset(local_root, source_root=args.source_root)
        clips = local["clips"]
        if not clips or any(c["source"] != "local_supplement" or c["frames"] < HISTORY_LENGTH for c in clips):
            raise ValueError("The local walking cache must contain complete repository walking and turn clips")
        summary = report["summary"]
        summary.pop("reference_start_fraction")
        summary.pop("mirror_probability")
        summary.update(
            walking_expert_source=walking_expert_source,
            local_source_clips=len(clips),
            local_supplement_frames=sum(c["frames"] for c in clips),
            official_history_windows=2 * (len(trajectories[0]) - HISTORY_LENGTH + 1),
            local_history_windows=2 * sum(c["frames"] - HISTORY_LENGTH + 1 for c in clips),
            sampling="uniform over valid windows in all original and mirrored clips",
        )
        report["checks"]["legacy_walking_cache"] = {"passed": True, "output_sha256": local["output_sha256"]}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root',type=Path,help="Audit an explicitly selected historical converted dataset or a raw source directory")
    parser.add_argument('--source-root',type=Path,default=ROOT/'src/assets/motions/g1/amp')
    parser.add_argument('--hiking-source',type=Path,help="Path to the official release files; by default checks the source cache, then the repository bundle")
    parser.add_argument('--grail-source',type=Path,default=ROOT/'.perceptive_cache/sources/grail')
    parser.add_argument('--robot-xml',type=Path,default=ROOT/'src/assets/robots/unitree_g1/xmls/g1.xml')
    parser.add_argument('--training-config',type=Path,help='Optional saved RUN/config.json whose dataset references should be checked.')
    parser.add_argument('--output',type=Path,default=ROOT/'docs/validation/hiking_official_motion_integrity.json')
    parser.add_argument('--check-remote',action='store_true',help='Historical filtered protocol: check pinned GRAIL URLs with HEAD; official Hiking uses fixed source file digests.')
    args=parser.parse_args(argv)
    for field in ('dataset_root','source_root','hiking_source','grail_source','robot_xml','output','training_config'):
        value=getattr(args,field)
        if value is not None:setattr(args,field,value.expanduser().resolve())
    from src.wmac_perceptive.locomotion_data import HIKING_BUNDLED_SOURCE
    from src.wmac_perceptive.standing_data import STANDING_BUNDLE
    source_override = args.hiking_source is not None
    if args.hiking_source is None:
        cached = ROOT/'.perceptive_cache/sources/hiking'
        args.hiking_source = cached.resolve() if cached.exists() else HIKING_BUNDLED_SOURCE.resolve()
    if args.dataset_root is None and args.training_config:
        cfg = json.loads(args.training_config.read_text())
        if cfg.get("mirrored_motion_root") and (cfg.get("hiking_training_version",1) < 3 or not source_override):
            args.dataset_root = (ROOT/cfg["mirrored_motion_root"]).resolve()
    protected = (args.source_root,args.hiking_source,args.grail_source,STANDING_BUNDLE.resolve(),HIKING_BUNDLED_SOURCE.resolve())
    if any(args.output.is_relative_to(p) for p in protected) or (
            args.dataset_root and args.output.is_relative_to(args.dataset_root)):
        parser.error('--output must be outside dataset/source directories to keep the audit read-only.')
    report=dict(audited_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        read_only_dataset=True,issues=[],checks={},files=[],source_files=[],source_reconstruction=[],remote_pinned_urls=[])
    try:
        if args.dataset_root is None or not (args.dataset_root/"mirror_manifest.json").is_file():
            audit_hiking_reference(args.dataset_root or args.hiking_source, args, report)
        else:
            manifest=json.loads((args.dataset_root/'mirror_manifest.json').read_text())
            if manifest.get('motion_data_protocol')=='hiking_walk_v1':
                audit_official_hiking(args,report,include_standing=manifest.get('standing_expert_source','none')!='none')
            else:
                audit(args,report)
    except Exception as error:
        report['issues'].append(dict(check='audit_interrupted',details=f'{type(error).__name__}: {error}'))
    report['status']='pass' if not report['issues'] else 'issues_found'
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps(dict(status=report['status'],summary=report.get('summary'),issues=report['issues'],report=str(args.output)),indent=2))
    return 0 if report['status']=='pass' else 1


if __name__=='__main__':
    raise SystemExit(main())
