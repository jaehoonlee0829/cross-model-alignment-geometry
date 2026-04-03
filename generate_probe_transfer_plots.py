"""
generate_probe_transfer_plots.py
Generates separate plots for within-family and cross-architecture
next-token probe transfer experiments.

These were previously combined into a single side-by-side figure, but the
two experiments use different methodologies (shared tokenizer vs matched-token
vocabulary) making a direct visual comparison misleading.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
import os

os.makedirs('outputs/plots', exist_ok=True)
plt.rcParams.update({'font.size': 11, 'figure.dpi': 150})

def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


# --- Plot 1: Within-family (Llama-1B → Llama-3B, shared tokenizer) ---

eval_c = load_csv('outputs/eval_c/probing/probe_transfer_results.csv')

fig, ax = plt.subplots(figsize=(10, 7))

baseline = next(r for r in eval_c if r['method'] == 'baseline_model_a')
oracle = next(r for r in eval_c if r['method'] == 'oracle_model_b')

rank_results = [(int(r['rank']), float(r['top1_accuracy']))
                for r in eval_c if r['method'].startswith('low_rank')]
rank_results.sort()
ranks, accs = zip(*rank_results)
ax.plot(ranks, accs, marker='o', linewidth=2, color='#2196F3', label='Transferred probe (low-rank)')

ridge = next(r for r in eval_c if r['method'] == 'ridge')
ax.scatter([512], [float(ridge['top1_accuracy'])], marker='D', s=100,
           color='#4CAF50', zorder=5, label=f"Ridge ({float(ridge['top1_accuracy']):.1%})")

ax.axhline(y=float(baseline['top1_accuracy']), color='blue', linestyle='--',
           alpha=0.5, label=f"Source native — Llama-1B ({float(baseline['top1_accuracy']):.1%})")
ax.axhline(y=float(oracle['top1_accuracy']), color='green', linestyle='--',
           alpha=0.5, label=f"Target oracle — Llama-3B ({float(oracle['top1_accuracy']):.1%})")

ax.set_xlabel('Alignment Rank', fontsize=12)
ax.set_ylabel('Top-1 Accuracy (next-token prediction)', fontsize=12)
ax.set_title('Within-Family NTP Probe Transfer: Llama-1B → Llama-3B\n'
             '(shared tokenizer, full vocabulary)', fontsize=12)
ax.set_xscale('log', base=2)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig('outputs/plots/probe_transfer_within_family.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: probe_transfer_within_family.png')


# --- Plot 2: Cross-architecture (Gemma-2B → Qwen-1.5B, matched-token vocab) ---

matched = load_csv('outputs/phase_b/probing/matched_token_probe_results.csv')

fig, ax = plt.subplots(figsize=(10, 7))

source = next(r for r in matched if r['method'] == 'source_native')
oracle = next(r for r in matched if r['method'] == 'target_oracle')
cross_oracle = next(r for r in matched if r['method'] == 'cross_model_oracle')

rank_results = [(int(r['rank']), float(r['top1']))
                for r in matched if r['method'].startswith('low_rank')]
rank_results.sort()
ranks, accs = zip(*rank_results)
ax.plot(ranks, accs, marker='o', linewidth=2, color='#FF5722', label='Transferred probe (low-rank)')

ridge = next(r for r in matched if r['method'] == 'ridge')
ax.scatter([512], [float(ridge['top1'])], marker='D', s=100,
           color='#4CAF50', zorder=5, label=f"Ridge ({float(ridge['top1']):.1%})")

ax.axhline(y=float(cross_oracle['top1']), color='red', linestyle='--',
           alpha=0.5, label=f"Cross-model oracle ceiling ({float(cross_oracle['top1']):.1%})")
ax.axhline(y=float(source['top1']), color='blue', linestyle=':',
           alpha=0.4, label=f"Source native — Gemma ({float(source['top1']):.1%})")
ax.axhline(y=float(oracle['top1']), color='green', linestyle=':',
           alpha=0.4, label=f"Target oracle — Qwen ({float(oracle['top1']):.1%})")

ax.set_xlabel('Alignment Rank', fontsize=12)
ax.set_ylabel('Top-1 Accuracy (next-token prediction)', fontsize=12)
ax.set_title('Cross-Architecture NTP Probe Transfer: Gemma-2B → Qwen-1.5B\n'
             '(matched-token vocabulary, top-500 shared classes)', fontsize=12)
ax.set_xscale('log', base=2)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig('outputs/plots/probe_transfer_cross_arch.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: probe_transfer_cross_arch.png')
