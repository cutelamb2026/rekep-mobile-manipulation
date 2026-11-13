#!/usr/bin/env python3
"""Conservative source-file secret scan. Prints paths/line numbers only, never values."""
import pathlib,re,sys
if len(sys.argv)!=2:raise SystemExit('Usage: python 07_secret_scan.py PATH')
root=pathlib.Path(sys.argv[1]);found=0
patterns=[re.compile(r'sk-[A-Za-z0-9_-]{12,}'),re.compile(r'(?i)(OPENAI_API_KEY|AWS_SECRET_ACCESS_KEY|api[_-]?key|secret[_-]?token)\s*[=:]\s*[\'\"]?\S{10,}')]
for p in ([root] if root.is_file() else root.rglob('*')):
 if not p.is_file() or p.stat().st_size>4_000_000 or '.git' in p.parts:continue
 try:lines=p.read_text(encoding='utf-8').splitlines()
 except (UnicodeError,OSError):continue
 for i,line in enumerate(lines,1):
  if any(pattern.search(line) for pattern in patterns):
   print(f'POSSIBLE_SECRET file={p} line={i}');found+=1
print(f'possible_secret_lines={found}; scanner is not a substitute for manual review')
sys.exit(1 if found else 0)
