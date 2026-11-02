#!/usr/bin/env python3
import hashlib, json, sys
from pathlib import Path
root=Path(sys.argv[1]); manifest=json.loads((root/'manifest.json').read_text()); results=[]
for f in manifest['files']:
    path=root/'model'/f['name']
    if not path.is_file(): results.append({'name':f['name'],'error':'missing'}); continue
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(16*1024*1024),b''): digest.update(chunk)
    actual=digest.hexdigest(); okay=(f['size'] is None or path.stat().st_size==f['size']) and (f['sha256'] is None or actual==f['sha256'])
    results.append({'name':f['name'],'size':path.stat().st_size,'sha256':actual,'expected_sha256':f['sha256'],'ok':okay})
    print(f['name'], 'OK' if okay else 'FAILED',flush=True)
report={'repo':manifest['repo'],'revision':manifest['revision'],'files':results,'ok':all(x.get('ok',False) for x in results)}
(root/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
sys.exit(0 if report['ok'] else 1)
