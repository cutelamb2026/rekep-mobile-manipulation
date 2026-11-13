#!/usr/bin/env python3
"""Structural validation of ReKep constraint program. Does NOT prove task relevance or success."""
import argparse,json,pathlib,sys
p=argparse.ArgumentParser();p.add_argument('--program_dir',required=True);a=p.parse_args()
d=pathlib.Path(a.program_dir);meta=d/'metadata.json'
if not meta.exists():raise SystemExit('MISSING metadata.json')
m=json.loads(meta.read_text(encoding='utf-8'))
required=['num_keypoints','num_stages','init_keypoint_positions','grasp_keypoints','release_keypoints']
for key in required:
 if key not in m:raise SystemExit('MISSING metadata key '+key)
k,n=int(m['num_keypoints']),int(m['num_stages'])
if n<1 or k<1:raise SystemExit('Invalid stage/keypoint counts')
if len(m['init_keypoint_positions'])!=k:raise SystemExit('Keypoint positions length mismatch')
for key in ['grasp_keypoints','release_keypoints']:
 if len(m[key])!=n:raise SystemExit(key+' length != num_stages')
print('Program',d,'stages',n,'keypoints',k)
for stage in range(1,n+1):
 for typ in ['subgoal','path']:
  pth=d/f'stage{stage}_{typ}_constraints.txt'
  print(f'stage{stage}_{typ}:', 'present' if pth.exists() else 'absent (zero constraints if loader allows)')
print('STRUCTURE OK; not a semantic/physical feasibility check')
