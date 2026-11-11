import pandas as pd
from pathlib import Path

hists = {
    'initial (LN)': Path('reports/training/tabular_resnet_history.csv'),
    'ablation_norm (BN)': Path('reports/training/tabular_resnet_ablation_norm_history.csv'),
    'tuned (LN)': Path('reports/training/tabular_resnet_tuned_history.csv'),
}

for name, p in hists.items():
    h = pd.read_csv(p)
    h1 = h[h['epoch'] > 3]
    collapses = h1[h1['balanced_accuracy'] < 0.90]
    print(f'{name}: epochs={len(h)}, best_ba={h.balanced_accuracy.max():.6f} (epoch {h.balanced_accuracy.idxmax()+1})')
    print(f'  epochs>3 with BA<0.90: {len(collapses)}')
    if len(collapses):
        print(f'  collapse epochs: {list(collapses.epoch.astype(int))}')
        print(f'  BA values: {[f"{x:.3f}" for x in collapses.balanced_accuracy.values]}')
    print(f'  lowest BA after epoch 3: {h1.balanced_accuracy.min():.4f}')
    print()

print('BEST MODEL PICK ORDER (balanced accuracy desc):')
results = [(name, pd.read_csv(p).balanced_accuracy.max()) for name, p in hists.items()]
results.sort(key=lambda x: -x[1])
for n, ba in results:
    print(f'  {n}: {ba:.6f}')
print()
print('TUNED - INITIAL delta BA:', f'{results[0][1] - results[1][1]:.6f}')
print('INITIAL (LN) - ABLATION (BN) effect BA (LN_vs_BN single factor):',
      f'{results[1][1] - results[2][1]:.6f}')
