#!/usr/bin/env python3
"""Read an existing ReKep scene JSON; list object registry info, no simulation startup."""
import argparse,json,pathlib
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);a=p.parse_args()
s=pathlib.Path(a.scene);d=json.loads(s.read_text(encoding='utf-8'))
info=d.get('objects_info',{}).get('init_info',{})
reg=d.get('state',{}).get('object_registry',{})
print('scene:',s,'registered:',len(reg),'initialized:',len(info))
for name,props in info.items():
 args=props.get('args',{});pose=reg.get(name,{}).get('root_link',{})
 if name.startswith(('floor','ceiling','wall')):continue
 print(f'{name:35s} category={str(args.get("category","")):18s} model={str(args.get("model","")):14s} pos={pose.get("pos",None)}')
