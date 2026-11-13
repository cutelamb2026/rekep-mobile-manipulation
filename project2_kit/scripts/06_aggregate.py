#!/usr/bin/env python3
"""Aggregate labelled episodes only. Do not fabricate outcomes or claim success from exit=0."""
import argparse,collections,csv,json,pathlib,math
p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--out',required=True);a=p.parse_args()
with open(a.input,encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
valid=[];unlabelled=0;ids=set()
for r in rows:
 if not any((v or '').strip() for v in r.values()):continue
 rid=(r.get('run_id') or '').strip()
 if not rid:raise SystemExit('Non-empty row without run_id')
 if rid in ids:raise SystemExit('Duplicate run_id: '+rid)
 ids.add(rid)
 raw=(r.get('success') or '').strip().lower()
 if raw in ('','unknown','na','none'):
  unlabelled+=1;continue
 if raw not in ('0','1'):raise SystemExit('success must be 0/1/blank; got '+repr(raw)+' run_id='+rid)
 r['success_val']=int(raw);valid.append(r)
def summary(items):
 n=len(items);k=sum(r['success_val'] for r in items)
 if not n:return {'n':0,'successes':0,'success_rate':None,'wilson_95ci':None}
 ph=k/n;z=1.96;d=1+z*z/n;c=(ph+z*z/(2*n))/d
 h=z*math.sqrt(ph*(1-ph)/n+z*z/(4*n*n))/d
 return {'n':n,'successes':k,'success_rate':round(ph,5),'wilson_95ci':[round(max(0,c-h),5),round(min(1,c+h),5)]}
grp=collections.defaultdict(list);fail=collections.Counter()
for r in valid:
 grp[(r.get('scenario') or 'UNSPECIFIED').strip()].append(r)
 if r['success_val']==0:fail[(r.get('failure_code') or 'unclassified').strip() or 'unclassified']+=1
out={'overall':summary(valid),'by_scenario':{key:summary(value) for key,value in sorted(grp.items())},'failure_codes':dict(fail),'unlabelled_rows_excluded':unlabelled,'note':'Only success=0 or success=1 counted. 95% Wilson intervals (approx).'}
pout=pathlib.Path(a.out);pout.parent.mkdir(parents=True,exist_ok=True);pout.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(out,ensure_ascii=False,indent=2))
