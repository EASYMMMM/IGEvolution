"""Use nondegenerate binomial intervals for rare early terminations."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    out = parser.parse_args().output.resolve()
    result = json.loads((out / 'results.json').read_text())
    rows = result['closed_loop']
    for row in rows:
        p, n, z = row['failed'], row['n_episodes'], 1.959963984540054
        center = (p + z*z/(2*n)) / (1 + z*z/n)
        radius = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
        row['failed_ci_low'] = max(0., float(center-radius))
        row['failed_ci_high'] = min(1., float(center+radius))
        row['failed_ci_method'] = 'Wilson score, 95%; episodes treated as independent'
    result['caveats'] = [s for s in result['caveats'] if not s.startswith('Rare termination')]
    result['caveats'].append('Rare termination rates use Wilson intervals; paired bootstrap differences are exploratory and cannot establish equivalent failure risk.')
    result['evaluator_source_sha256'].update({p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in Path(__file__).parent.glob('*.py')})
    (out/'results.json').write_text(json.dumps(result, indent=2))
    with (out/'closed_loop_summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    names = ['Intact', 'Zero z', 'Mean z', 'Swap z', 'Delay 240 ms',
             'Swap HMI inputs', 'Last 5 frames', 'Intact repeat']
    colors = ['#1f7a8c','#e15759','#f28e2b','#9467bd','#4e79a7','#b07aa1','#76b7b2','#8b8b8b']
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10,
                        'axes.spines.top':False, 'axes.spines.right':False,
                        'axes.titleweight':'bold'})
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, metric, title, scale in zip(axes.flat,
            ['failed', 'duration', 'position_rmse', 'force_mean'],
            ['Early termination rate (%) | Wilson 95% CI', 'First-episode duration (s)',
             'Position tracking RMSE while alive (m)', 'Mean interaction force while alive (N)'],
            [100, 1, 1, 1]):
        values = np.array([r[metric] for r in rows])*scale
        low = np.array([r[metric+'_ci_low'] for r in rows])*scale
        high = np.array([r[metric+'_ci_high'] for r in rows])*scale
        ax.bar(np.arange(8), values, color=colors,
               yerr=np.maximum(0, np.array([values-low, high-values])), capsize=3)
        ax.set_xticks(range(8))
        ax.set_xticklabels(names, rotation=28, ha='right', fontsize=8)
        ax.set_title(title)
    fig.suptitle('Closed-loop causal interventions | 192 matched episodes per condition | 75 s cap', fontsize=15)
    fig.tight_layout()
    for extension in ('png', 'svg'):
        fig.savefig(out/'figures'/('05_closed_loop.'+extension), dpi=180,
                    bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('SUMMARY INTERVALS COMPLETE', flush=True)


if __name__ == '__main__':
    main()
