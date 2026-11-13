#!/usr/bin/env python3
"""Create a *planned* 12-layout x 8-placement matrix; it is NOT simulated data."""
import argparse,csv,pathlib
p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
layouts=['source_center_target_front','source_left_target_front','source_right_target_front','source_center_target_left','source_center_target_right','source_rear_target_front']
states=['nominal','dx_plus','dx_minus','dy_plus','dy_minus','yaw_plus_15','yaw_minus_15','near_edge']
out=pathlib.Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
with out.open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.writer(f);w.writerow(['scenario','layout_id','layout_description','placement_id','placement_description','scene_file','scene_ready','validated','notes'])
 for scenario in ['logistics','retail']:
  for i,lay in enumerate(layouts,1):
   for j,state in enumerate(states,1):
    w.writerow([scenario,f'{scenario[:3].upper()}{i:02d}',lay,f'P{j:02d}',state,'',0,0,'PROPOSED; must build and validate actual asset placement'])
print('Wrote',out,'planned_combinations=96; validated=0')
