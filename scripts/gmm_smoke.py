import sys, time
sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts')
import numpy as np, polars as pl
from sota_eval_kronos import dip_challengers, _fit_gmm, TAUS

df = pl.read_parquet(r'data\raw\sources\btcusdt_1d_deep.parquet').sort('event_time')
px = df['close'].to_numpy()
rets_all = np.diff(px) / px[:-1]
i = rets_all.size - 50  # mid-tail origin
rets = rets_all[i-250:i]; rets_long = rets_all[i-750:i]; y = rets_all[i]
t0 = time.time()
c, q = dip_challengers(rets, rets_long, y)
dt = time.time() - t0
print('dip_gmm_k crps:', c['dip_gmm_k'])
print('dip_gmm_k q5/50/95:', q['dip_gmm_k'])
w, mu, sig = _fit_gmm(rets_long)
print('K=%d w=%s' % (w.size, np.round(w,3)))
print('all finite:', np.isfinite(list(c.values())).all() if hasattr(c.values(),'all') else all(np.isfinite(v) for v in c.values()))
print('elapsed %.2fs for %d challengers' % (dt, len(c)))
