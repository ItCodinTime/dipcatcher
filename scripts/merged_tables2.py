import json
for tag in ['d1_merged', 'h4_merged']:
    r = json.load(open(rf'D:\dipcatcher\.dsh-24x7\eval-full\{tag}.json'))
    print('=====', tag, '| n_complete =', r['n_complete'])
    print('-- pooled CRPS --')
    for k, v in sorted(r['mean_crps_pooled'].items(), key=lambda kv: kv[1]):
        flag = '*' if r['mcs_included'].get(k) else ' '
        print('   ', flag, k.ljust(20), round(v, 5))
    print('-- per-asset CRPS --')
    for asset, d in r['mean_crps_per_asset'].items():
        tgt = {k: d[k] for k in ('kronos_small','chronos2','bolt_small','timesfm') if k in d}
        dips = {k: v for k, v in d.items() if k.startswith('dip_')}
        best_t = min(tgt.values()); worst_d = max(dips.values()); best_d = min(dips.values())
        sweep = worst_d < best_t
        print(f'    {asset}: best_target={best_t:.5f} | worst_dip={worst_d:.5f} best_dip={best_d:.5f} | all-dips-beat-all-targets: {sweep}')
    print('-- pinball keys sample --')
    k0 = next(iter(r['mean_pinball_pooled']))
    print('   pinball entry for', k0, '=', r['mean_pinball_pooled'][k0])
