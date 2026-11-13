#!/usr/bin/env python3
"""Make a pose-only copy of an existing object in ReKep/OmniGibson scene JSON.
Does not create new assets; must be validated by simulation.
"""
import argparse,json,math,pathlib
p=argparse.ArgumentParser()
for arg in ['source','object','out']:p.add_argument('--'+arg,required=True)
for arg in ['dx','dy','dz','yaw_deg']:p.add_argument('--'+arg,type=float,default=0)
a=p.parse_args()
src=pathlib.Path(a.source).resolve();out=pathlib.Path(a.out).resolve()
if src==out:raise SystemExit('Refusing to overwrite source scene')
d=json.loads(src.read_text(encoding='utf-8'))
if a.object not in d.get('state',{}).get('object_registry',{}):
 raise SystemExit('Object name not found in state.object_registry; inspect scene JSON')
if a.object not in d.get('objects_info',{}).get('init_info',{}):
 raise SystemExit('Object absent from objects_info.init_info; inspect scene JSON')
pose=d['state']['object_registry'][a.object]['root_link']
pose['pos']=[x+y for x,y in zip(pose['pos'],[a.dx,a.dy,a.dz])]
if a.yaw_deg:
 # Existing scene orientations use [x, y, z, w]. Left-multiply world-Z yaw.
 x,y,z,w=pose['ori'];t=math.radians(a.yaw_deg)/2;sz,cz=math.sin(t),math.cos(t)
 q=[cz*x-sz*y,sz*x+cz*y,cz*z+sz*w,cz*w-sz*z]
 norm=math.sqrt(sum(v*v for v in q));pose['ori']=[v/norm for v in q]
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
print('Generated variant',out,'object=',a.object,'(NOT simulator-validated)')
