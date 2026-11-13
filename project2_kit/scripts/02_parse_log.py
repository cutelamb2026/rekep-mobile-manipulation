#!/usr/bin/env python3
"""Extract *observed* ReKep solver metrics. Does not infer episode success."""
import argparse,json,re,pathlib
p=argparse.ArgumentParser()
p.add_argument('--log',required=True);p.add_argument('--out',required=True)
a=p.parse_args()
s=pathlib.Path(a.log).read_text(encoding='utf-8',errors='replace')
ansi=re.compile(r'\x1b\[[0-9;]*[A-Za-z]')
s=ansi.sub('',s)
def getall(pattern):return re.findall(pattern,s,re.I|re.M)
def numeric(name):
 values=[]
 for raw in getall(r'^\s*#\s*'+re.escape(name)+r'\s*:\s*([^\r\n]+)'):
  val=re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?',raw)
  if val:
   try:values.append(float(val[0]))
   except ValueError:pass
 return values
out={
 'log_path':str(pathlib.Path(a.log).resolve()),
 'num_solver_sections':len(getall(r'^\s*#\s*type\s*:\s*(?:subgoal_solver|path_solver)')),
 'stages_reported':numeric('stage'),
 'solve_times_s':numeric('solve_time'),
 'ik_feasible_numeric':numeric('ik_feasible'),
 'path_constraint_cost':numeric('path_constraint_cost'),
 'total_cost':numeric('total_cost'),
 'osc_tracking_warnings':len(getall(r'OSC pose not reached')),
 'workspace_clipping_warnings':len(getall(r'Target position is out of bounds')),
 'backtrack_events':len(getall(r'backtrack to stage')),
 'reported_video_paths':getall(r'Video saved to\s+([^\s]+)'),
 'task_success':None,
 'note':'Run metrics only. Human or simulator state check must determine task success.'
}
pout=pathlib.Path(a.out);pout.parent.mkdir(parents=True,exist_ok=True)
pout.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print('saved',pout, 'solver_sections',out['num_solver_sections'])
