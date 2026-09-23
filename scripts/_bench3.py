import subprocess, sys
bars = ['data/raw/sources/btcusdt_1d.parquet','data/raw/sources/ethusdt_1d.parquet','data/raw/sources/solusdt_1d.parquet']
sys.exit(subprocess.call(
 [r'D:\bench-qlib\Scripts\python.exe','scripts/incumbent_bench_vectorbt.py',
  '--bars',*bars,'--engine','fast',
  '--out',r'.dsh-24x7\evidence-incumbent-vectorbt-fast-v2.json'],
 cwd=r'D:\dipcatcher'))
