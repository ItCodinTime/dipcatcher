Set-Location D:\dipcatcher
$bars = @(
 'data/raw/sources/adausdt_1d.parquet','data/raw/sources/avaxusdt_1d.parquet',
 'data/raw/sources/bnbusdt_1d.parquet','data/raw/sources/btcusdt_1d.parquet',
 'data/raw/sources/dogeusdt_1d.parquet','data/raw/sources/ethusdt_1d.parquet',
 'data/raw/sources/linkusdt_1d.parquet','data/raw/sources/ltcusdt_1d.parquet',
 'data/raw/sources/solusdt_1d.parquet','data/raw/sources/trxusdt_1d.parquet',
 'data/raw/sources/xrpusdt_1d.parquet')
D:\bench-qlib\Scripts\python.exe scripts\incumbent_bench_vectorbt.py --bars $bars --engine fast --out .dsh-24x7\evidence-incumbent-vectorbt-fast-11a-v2.json *> D:\dipcatcher\_bench11.txt
Get-Content D:\dipcatcher\_bench11.txt
