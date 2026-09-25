import sys
import time

sys.path.insert(0, 'src')
sys.path.insert(0, 'scripts')
import numpy as np
import polars as pl
from sota_eval_kronos import TAUS, _fit_skt, _skt_quantiles, dip_challengers

df = pl.read_parquet(r'data\raw\sources\btcusdt_1d_deep.parquet').sort('event_time')
px = df['close'].to_numpy()
rets_all = np.diff(px) / px[:-1]
i = rets_all.size - 50
rets = rets_all[i-250:i]
rets_long = rets_all[i-750:i]
y = rets_all[i]
t0 = time.time()
xi, om, al, nu = _fit_skt(rets_long)
print(f'skt fit: xi={xi:.5f} om={om:.5f} alpha={al:.2f} nu={nu:.1f} ({time.time()-t0:.2f}s)')
q = _skt_quantiles(xi, om, al, nu, np.asarray(TAUS))
print('quantiles:', q)
c, qq = dip_challengers(rets, rets_long, y)
print('dip_skt crps:', c['dip_skt'], '| monotone:', np.all(np.diff(qq['dip_skt']) > 0))
