"""Offline rebase of one checked trial onto a freshly captured scene (either arm; copy of 1107 left-only script)."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'deploy/policy_control/tools'))
sys.path.insert(0,str(ROOT/'deploy/policy_control'))
import plan_home_path as P
import home_path_world as W
import stationary_tracking as cli
from policy_control import stationary_tracking as S
p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--trial',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
plan=json.loads(a.plan.read_text());snap=json.loads(a.snapshot.read_text());scene=snap['scene'];trial=next(t for t in plan['trials'] if t['id']==a.trial)
# The mirrored target and fixed hand are kept; any hand change requires new IK.
assert all(abs(scene[k]-v)<.005 for k,v in plan['fixed_hand'].items()), 'fixed hand changed; rerun prepare/IK'
old=np.asarray(trial['approach_frames']);q=np.asarray(snap['q']);limits=W.load_profile_limits(Path(plan['inputs']['profile']['path']));joints=S.arm_joint_names(plan["side"]);lo=np.array([limits[j][0] for j in joints]);hi=np.array([limits[j][1] for j in joints]);world=W.build_world(W.WorldSpec(urdf=Path(plan['inputs']['urdf']['path']),env_yaml=Path(plan['inputs']['env_yaml']['path']),side=plan["side"]));checker=P.Checker(world,[scene],.02,q,(lo,hi));checker.lenient=False
ramp=P.time_parametrize(np.stack([q,old[0]]),vmax=.05,dt=.01,ramp=1.)
approach=np.vstack([ramp[:-1],old]);full=np.vstack([approach[:-1],np.asarray(trial['frames'])]);report=checker.check_path(full)
assert report['ok'],str(report['fails'][:3])
trial['approach_start_q']=q.tolist();trial['approach_frames']=approach.tolist();trial['return_frames']=np.vstack([np.asarray(trial['frames'])[::-1],approach[-2::-1]]).tolist();trial['approach_collision']={'min_clear_m':float(report['min_clear']),'samples':report['samples'],'rechecked_after_abort':True};plan['trials']=[trial];plan['scene']=scene
inputs={k:Path(v['path']) for k,v in plan['inputs'].items()};inputs['scene']=a.snapshot
m=S.new_run(a.out,inputs,{'mode':'prepare','source_plan':str(a.plan),'reason':'rebase from measured pose','commands_sent':0},repo=cli.ROOT);plan['inputs']=m['inputs'];cli.dump(a.out/'plan.json',plan);(a.out/'events.jsonl').touch();S.write_report(a.out,{'eligible':False,'pass':False,'commands_sent':0,'excluded_reasons':['preparation_only'],'recheck':trial['approach_collision']},[],[]);print('rebase checked',trial['approach_collision'])
