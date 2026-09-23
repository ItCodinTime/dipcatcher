import numpy as np, json
for tag in ['h4_ethusdt_4h_deep', 'h4_xrpusdt_4h']:
    z = np.load(rf'D:\dipcatcher\.dsh-24x7\eval-full\{tag}.losses.npz', allow_pickle=False)
    names = [str(x) for x in z['model_names']]
    M = z['crps_matrix']
    print('###', tag, M.shape)
    for j, m in enumerate(names):
        nn = int(np.isnan(M[:, j]).sum())
        if nn: print('   NaN', m, nn)
    P = z['pinball_cube']
    for j, m in enumerate(names):
        nn = int(np.isnan(P[:, j, :]).sum())
        if nn: print('   pinball-NaN', m, nn)
