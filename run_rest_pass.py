import glob, os, subprocess, time
from concurrent.futures import ThreadPoolExecutor

EV = r'D:\evalenv\Scripts\python.exe'
CWD = r'D:\dipcatcher'
SHARDS = sorted(
    glob.glob(r'D:\dipcatcher\.dsh-24x7\eval-full\d1fix_*.losses.npz') +
    glob.glob(r'D:\dipcatcher\.dsh-24x7\eval-full\h4f_*.losses.npz') +
    glob.glob(r'D:\dipcatcher\.dsh-24x7\eval-full\h4fix_*.losses.npz') +
    glob.glob(r'D:\dipcatcher\.dsh-24x7\eval-full\seed11_*.losses.npz') +
    glob.glob(r'D:\dipcatcher\.dsh-24x7\eval-full\seed23_*.losses.npz')
)
TASKS = [(s, m) for s in SHARDS for m in ('dip_qar', 'dip_conf_t', 'dip_regime')]
print('%d tasks over %d shards' % (len(TASKS), len(SHARDS)), flush=True)

def col(task):
    shard, model = task
    tag = model.replace('dip_', '')
    out = shard.replace('.losses.npz', '.%scol.npz' % tag)
    if os.path.exists(out):
        return task, 'col-skip'
    r = subprocess.run([EV, 'scripts\\_challenger_col.py', '--shard', shard,
                        '--model', model, '--bars-root', r'data\raw\sources',
                        '--out', out], cwd=CWD, capture_output=True, text=True)
    return task, ('ok' if r.returncode == 0 else 'FAIL ' + r.stderr[-300:])

t0 = time.time()
fails = 0
with ThreadPoolExecutor(max_workers=3) as ex:
    for (shard, model), st in ex.map(col, TASKS):
        if 'FAIL' in st:
            fails += 1
            print('[%6.0fs] %s %s: %s' % (time.time()-t0, os.path.basename(shard), model, st), flush=True)
        else:
            print('[%6.0fs] %s %s: %s' % (time.time()-t0, os.path.basename(shard), model, st), flush=True)
print('DONE fails=%d' % fails, flush=True)
