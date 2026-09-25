import glob
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

EV = r'D:\evalenv\Scripts\python.exe'
CWD = r'D:\dipcatcher'
SHARDS = sorted(glob.glob(r'D:\dipcatcher\.dsh-24x7\eval-full\*.withgmm.npz'))
print(f'{len(SHARDS)} withgmm shards', flush=True)

def col(shard):
    out = shard.replace('.withgmm.npz', '.withgmm.sktcol.npz')
    if os.path.exists(out):
        return shard, 'col-skip'
    r = subprocess.run([EV, 'scripts\\_challenger_col.py', '--shard', shard,
                        '--model', 'dip_skt',
                        '--bars-root', r'data\raw\sources', '--out', out],
                       cwd=CWD, capture_output=True, text=True)
    return shard, ('col-ok' if r.returncode == 0 else 'col-FAIL ' + r.stderr[-300:])

def splice(shard):
    colf = shard.replace('.withgmm.npz', '.withgmm.sktcol.npz')
    r = subprocess.run([EV, 'scripts\\splice_challenger_column.py', '--pairs', shard, colf],
                       cwd=CWD, capture_output=True, text=True)
    return shard, ('splice-ok' if r.returncode == 0 else 'splice-FAIL ' + r.stderr[-300:])

t0 = time.time()
with ThreadPoolExecutor(max_workers=4) as ex:
    for shard, st in ex.map(col, SHARDS):
        print(f'[{time.time()-t0:6.0f}s] {os.path.basename(shard)}: {st}', flush=True)
with ThreadPoolExecutor(max_workers=8) as ex:
    for shard, st in ex.map(splice, SHARDS):
        print(f'[{time.time()-t0:6.0f}s] {os.path.basename(shard)}: {st}', flush=True)
print('DONE', flush=True)
