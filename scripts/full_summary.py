import glob
import json
import os

for f in sorted(glob.glob(r'D:\dipcatcher\.dsh-24x7\eval-full\d1_*.json')):
    r = json.load(open(f))
    tag = os.path.basename(f).replace('.json','').replace('d1_','')
    print('###', tag, 'n =', r['n_complete'], 'dropped =', r['n_dropped_incomplete'])
    for k, v in sorted(r['mean_crps_pooled'].items(), key=lambda kv: kv[1]):
        flag = '*' if r['mcs_included'].get(k) else ' '
        print('   ', flag, k.ljust(20), round(v, 5))
