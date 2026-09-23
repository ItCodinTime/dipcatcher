import json

for tag in ['d1_merged', 'h4_merged']:
    r = json.load(open(rf'D:\dipcatcher\.dsh-24x7\eval-full\{tag}.json'))
    print('=====', tag, '| n_complete =', r['n_complete'], '| assets:', r['n_rows'])
    print('-- pooled CRPS --')
    for k, v in sorted(r['mean_crps_pooled'].items(), key=lambda kv: kv[1]):
        flag = '*' if r['mcs_included'].get(k) else ' '
        print('   ', flag, k.ljust(20), round(v, 5))
    print('-- mean pinball (tau 0.05/0.5/0.95) --')
    for k, v in r['mean_pinball_pooled'].items():
        print('   ', k.ljust(20), [round(x,5) for x in v])
    print('-- per-asset CRPS --')
    for asset, d in r['mean_crps_per_asset'].items():
        best_t = min((d[k] for k in ('kronos_small','chronos2','bolt_small','timesfm')), default=float('nan'))
        best_d = min(d[k] for k in d if k.startswith('dip_'))
        print(f'    {asset}: best_target={best_t:.5f} best_dip={best_d:.5f} margin={(best_t-best_d)/best_t*100:.1f}%')
