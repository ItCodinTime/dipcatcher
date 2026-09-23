import json, glob, os
for f in sorted(glob.glob(r'D:\dipcatcher\.dsh-24x7\eval-full\seed*_*.json')):
    r = json.load(open(f))
    tag = os.path.basename(f).replace('.json','')
    print('###', tag, 'n =', r['n_complete'])
    for k, v in sorted(r['mean_crps_pooled'].items(), key=lambda kv: kv[1]):
        flag = '*' if r['mcs_included'].get(k) else ' '
        print('   ', flag, k.ljust(20), round(v, 5))
