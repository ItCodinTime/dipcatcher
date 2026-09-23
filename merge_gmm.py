import glob, subprocess
EV = r'D:\evalenv\Scripts\python.exe'
CWD = r'D:\dipcatcher'
for fam, pat in [
    ('d1fix', 'd1fix_*.withgmm.npz'),
    ('h4f',   'h4f_*.withgmm.npz'),
    ('h4fix', 'h4fix_*.withgmm.npz'),
]:
    parts = sorted(glob.glob('D:\\dipcatcher\\.dsh-24x7\\eval-full\\' + pat))
    out = '.dsh-24x7\\eval-full\\merge_%s_withgmm.json' % fam
    r = subprocess.run([EV, 'scripts\\sota_eval_kronos.py', '--merge-parts', *parts,
                        '--merge-out', out, '--n-boot', '2000'],
                       cwd=CWD, capture_output=True, text=True)
    print('=== %s (%d parts) rc=%d' % (fam, len(parts), r.returncode))
    print((r.stdout + r.stderr)[-3000:])
