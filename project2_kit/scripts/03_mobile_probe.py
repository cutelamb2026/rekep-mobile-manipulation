#!/usr/bin/env python3
"""Small, state-logging mobile base probe. Intended for a safe, verified free space."""
import argparse,json,pathlib,sys,numpy as np
pa=argparse.ArgumentParser()
pa.add_argument('--repo',required=True);pa.add_argument('--out',required=True)
pa.add_argument('--scene',default='configs/og_scene_file_pen.json')
pa.add_argument('--steps',type=int,default=10);pa.add_argument('--speed',type=float,default=0.10)
a=pa.parse_args()
if not 1 <= a.steps <= 30 or not 0 < a.speed <= 0.2:
 raise SystemExit('Safety guard: steps 1-30 and speed 0-0.2 only')
root=pathlib.Path(a.repo).resolve();sys.path.insert(0,str(root))
from utils import get_config
from environment import ReKepOGEnv
conf=get_config(config_path=str(root/'configs/config.yaml'))
env=ReKepOGEnv(conf['env'],str(root/a.scene),verbose=False)
env.reset()
robot=env.robot
if robot.action_dim < 2:raise RuntimeError('Unexpected action_dim; inspect base controller indexing')
def fmt(x):return np.asarray(x).astype(float).tolist()
def snapshot():
 base_pos,base_orn=robot.get_position_orientation()
 res={'base_position':fmt(base_pos),'base_orientation':fmt(base_orn)}
 try:res['ee_pose']=fmt(env.get_ee_pose())
 except Exception as e:res['ee_pose_error']=str(e)[:180]
 return res
snap={'pre':snapshot(),'action_dim':int(robot.action_dim),'control_assumption':'action[0] forward base velocity; confirm actual controller'}
for phase,sign in [('forward',1),('return',-1)]:
 action=np.zeros(robot.action_dim,dtype=float);action[0]=a.speed*sign
 for _ in range(a.steps): env.og_env.step(action)
 snap[phase]=snapshot()
try:
 cam=env.get_cam_obs()
 snap['cameras']={str(k):sorted(map(str,v.keys())) if isinstance(v,dict) else str(type(v)) for k,v in cam.items()}
except Exception as e:snap['camera_error']=str(e)[:300]
snap['base_motion_m']=float(np.linalg.norm(np.array(snap['forward']['base_position'])-np.array(snap['pre']['base_position'])))
out=pathlib.Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(snap,ensure_ascii=False,indent=2),encoding='utf-8')
print('Wrote',out,'base_motion_m=',snap['base_motion_m'])
if snap['base_motion_m'] < 0.005:
 print('WARNING: small movement. Validate action[0] mapping and control mode; do not claim success.')
