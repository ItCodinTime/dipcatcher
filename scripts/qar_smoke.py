import sys, time
sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts')
import numpy as np, polars as pl
from sota_eval_kronos import _qar_quantiles, dip_challengers

df = pl.read_parquet(r'data\raw\sources\btcusdt_1d_deep.parquet').sort('event_time')
px = df['close'].to_numpy()
rets_all = np.diff(px) / px[:-1]
i = rets_all.size - 50
rets = rets_all[i-250:i]; rets_long = rets_all[i-750:i]; y = rets_all[i]
t0 = time.time()
q = _qar_quantiles(rets_long)
print('qar levels:', np.round(q, 5), '(%.2fs)' % (time.time()-t0))
c, qq = dip_challengers(rets, rets_long, y)
print('dip_qar crps:', c['dip_qar'], '| q@TAUS:', qq['dip_qar'])
