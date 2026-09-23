import sys, time
sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts')
import numpy as np, polars as pl
from sota_eval_kronos import dip_challengers

df = pl.read_parquet(r'data\raw\sources\btcusdt_1d_deep.parquet').sort('event_time')
px = df['close'].to_numpy()
rets_all = np.diff(px) / px[:-1]
i = rets_all.size - 50
rets = rets_all[i-250:i]; rets_long = rets_all[i-750:i]; y = rets_all[i]
t0 = time.time()
c, q = dip_challengers(rets, rets_long, y)
print('elapsed %.1fs' % (time.time()-t0))
for k in sorted(c):
    mono = np.all(np.diff(q[k][np.isfinite(q[k])]) >= 0)
    print('%18s crps=%.6f mono=%s' % (k, c[k], mono))
