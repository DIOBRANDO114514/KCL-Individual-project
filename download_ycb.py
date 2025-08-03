import json, urllib.request, pathlib, time, socket, tqdm, os, sys

# ---------- Constant ----------
root = pathlib.Path('ycb_tgz')
root.mkdir(exist_ok=True)
socket.setdefaulttimeout(30)                 # Global timeout 30s

index_url = 'https://ycb-benchmarks.s3.amazonaws.com/data/objects.json'
no_rgb    = {                                # Officially there are really no RGB-highres for some of the objects.
    '027_skillet', '028_skillet_lid', '070-b_colored_wood_blocks'
}
max_retry = 5

# ---------- Getting a list of objects ----------
try:
    objects = json.loads(urllib.request.urlopen(index_url).read())['objects']
except Exception as e:
    sys.exit(f'Unavailable objects.json：{e}')

# ---------- Download ----------
failed = []
for obj in tqdm.tqdm(objects, desc='downloading'):
    if obj in no_rgb:         # Not at all. Just skip it.
        continue

    out_path = root / f'{obj}.tgz'
    if out_path.exists():
        continue              # Downloaded in full

    url = f'https://ycb-benchmarks.s3.amazonaws.com/data/berkeley/{obj}/{obj}_berkeley_rgb_highres.tgz'

    for attempt in range(1, max_retry + 1):
        try:
            req = urllib.request.Request(
                url, headers={'User-Agent': 'Mozilla/5.0'}
            )
            with urllib.request.urlopen(req) as resp, open(out_path, 'wb') as f:
                f.write(resp.read())         # Successful file write
            break                            

        except Exception as e:
            # Cleaning up half-documents
            if out_path.exists():
                try: out_path.unlink()
                except: pass

            if attempt == max_retry:
                failed.append(obj)
                tqdm.tqdm.write(f'Error: {obj} failed after {max_retry} tries: {e}')
            else:
                wait = 5 * attempt           # 5,10,15...seconds to evade
                tqdm.tqdm.write(f'Warning:  {obj} retry {attempt}/{max_retry} ({e}); wait {wait}s')
                time.sleep(wait)

# ---------- Results ----------
done  = len([p for p in root.glob('*.tgz') if p.stem not in no_rgb])
total = len(objects) - len(no_rgb)
print(f'\n✓ Successfully downloaded {done}/{total} objects (excluded 3 from 404)')

if failed:
    print('Still unsuccessful objects:', failed)
    print('You can keep the network stable and then re-run the script, it will only try these.')
else:
    print('All available objects have been downloaded!')
