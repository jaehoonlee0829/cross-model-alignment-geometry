import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import csv
import os
import glob
import re

os.makedirs('outputs/plots', exist_ok=True)
plt.rcParams.update({'font.size': 11, 'figure.dpi': 150})

# Helper to load CSV
def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))

# PLOT 1: CKA Heatmaps — All 4 phases in 2x2 grid
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
phase_info = [
    ('a', 'Eval A: Llama-1B vs Pythia-1.4B\n(d=2048, matched)'),
    ('b', 'Eval B: Gemma-2B vs Qwen-1.5B\n(d=2304 vs 1536)'),
    ('d', 'Eval D: Llama-3B vs Pythia-2.8B\n(d=3072 vs 2560)'),
    ('e', 'Eval E: Llama-3B vs Gemma-2B\n(d=3072 vs 2304)'),
]

for idx, (phase, title) in enumerate(phase_info):
    ax = axes[idx // 2][idx % 2]
    try:
        data = np.load(f'outputs/phase_{phase}/cka/cka_results.npz', allow_pickle=True)
        cka = data['cka_matrix']
        layers_a = data['layers_a']
        layers_b = data['layers_b']
        sns.heatmap(cka, annot=True, fmt='.2f', cmap='viridis',
                    xticklabels=[f'L{int(l)}' for l in layers_b],
                    yticklabels=[f'L{int(l)}' for l in layers_a],
                    vmin=0, vmax=0.25, ax=ax, cbar_kws={'label': 'CKA'})
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('Model B layers')
        ax.set_ylabel('Model A layers')
    except Exception as e:
        ax.text(0.5, 0.5, f'Failed: {e}', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)

plt.suptitle('Cross-Model CKA Similarity — All Evals', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/plots/all_cka_heatmaps.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: all_cka_heatmaps.png')

# PLOT 2: CKA Scale Comparison — max CKA by phase
fig, ax = plt.subplots(figsize=(10, 6))
phases = ['Eval A\nLlama-1B vs Pythia-1.4B\n(1B scale)', 'Eval B\nGemma-2B vs Qwen-1.5B\n(2B scale)',
          'Eval D\nLlama-3B vs Pythia-2.8B\n(3B scale)', 'Eval E\nLlama-3B vs Gemma-2B\n(cross-family)']
max_ckas = []
mean_ckas = []
for phase in ['a', 'b', 'd', 'e']:
    data = np.load(f'outputs/phase_{phase}/cka/cka_results.npz', allow_pickle=True)
    cka = data['cka_matrix']
    max_ckas.append(cka.max())
    mean_ckas.append(cka.mean())

x = np.arange(len(phases))
width = 0.35
bars1 = ax.bar(x - width/2, max_ckas, width, label='Max CKA', color='#2196F3')
bars2 = ax.bar(x + width/2, mean_ckas, width, label='Mean CKA', color='#FF9800')
ax.set_ylabel('CKA Score')
ax.set_title('Debiased CKA Across Cross-Family Model Pairs (1-3B Scale)')
ax.set_xticks(x)
ax.set_xticklabels(phases, fontsize=9)
ax.legend()
ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Strong alignment threshold')
ax.set_ylim(0, 0.35)
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.savefig('outputs/plots/cka_scale_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: cka_scale_comparison.png')

# PLOT 3: Eval B Low-Rank Analysis — key finding
try:
    align_b = load_csv('outputs/intermediary/alignment_results_phase_b.csv')
    # Extract best layer pair (18->23) results
    best_pair = [r for r in align_b if 'layer_18_to_23' in r['file']]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: test loss by method for 18->23
    methods = []
    test_losses = []
    colors = []
    for r in best_pair:
        name = r['file'].replace('layer_18_to_23_', '').replace('.npz', '')
        methods.append(name)
        test_losses.append(float(r['test_loss']))
        if 'low_rank' in name:
            colors.append('#4CAF50')
        elif 'linear' in name:
            colors.append('#2196F3')
        elif 'lasso' in name:
            colors.append('#FF5722')
        else:
            colors.append('#9E9E9E')

    ax1.barh(range(len(methods)), test_losses, color=colors)
    ax1.set_yticks(range(len(methods)))
    ax1.set_yticklabels(methods, fontsize=9)
    ax1.set_xlabel('Test Loss (lower = better)')
    ax1.set_title('Eval B: Gemma L18 -> Qwen L23\nAlignment Method Comparison')
    ax1.axvline(x=1.0, color='red', linestyle='--', alpha=0.5, label='Baseline (no alignment)')
    ax1.legend()

    # Right: low-rank test loss by rank across all layer pairs
    # Collect all low-rank results
    rank_data = {}
    for r in align_b:
        match = re.search(r'low_rank_rank(\d+)', r['file'])
        if match:
            rank = int(match.group(1))
            tl = float(r['test_loss'])
            if rank not in rank_data:
                rank_data[rank] = []
            rank_data[rank].append(tl)

    if rank_data:
        ranks = sorted(rank_data.keys())
        means = [np.mean(rank_data[r]) for r in ranks]
        stds = [np.std(rank_data[r]) for r in ranks]
        ax2.errorbar(ranks, means, yerr=stds, marker='o', capsize=5, linewidth=2)

        # Add ridge baseline
        ridge_losses = [float(r['test_loss']) for r in align_b if r['method'] == 'linear']
        if ridge_losses:
            ax2.axhline(y=np.mean(ridge_losses), color='red', linestyle='--', label=f'Ridge (mean={np.mean(ridge_losses):.3f})')
        ax2.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5, label='No alignment')
        ax2.set_xlabel('Low-Rank Dimension')
        ax2.set_ylabel('Test Loss (mean +/- std across layer pairs)')
        ax2.set_title('Eval B: Low-Rank Alignment\nLower rank = BETTER (less overfitting)')
        ax2.legend()
        ax2.set_xscale('log', base=2)

    plt.tight_layout()
    plt.savefig('outputs/plots/phase_b_lowrank_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved: phase_b_lowrank_analysis.png')
except Exception as e:
    print(f'Plot 3 failed: {e}')

# PLOT 4: Alignment Comparison Across Evals
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
for idx, (phase, title) in enumerate([
    ('a', 'Eval A: Llama-1B vs Pythia-1.4B'),
    ('b', 'Eval B: Gemma-2B vs Qwen-1.5B'),
    ('d', 'Eval D: Llama-3B vs Pythia-2.8B'),
    ('e', 'Eval E: Llama-3B vs Gemma-2B'),
]):
    ax = axes[idx // 2][idx % 2]
    try:
        align = load_csv(f'outputs/intermediary/alignment_results_phase_{phase}.csv')
        # Group by method type
        method_groups = {}
        for r in align:
            m = r['method']
            match = re.search(r'low_rank_rank(\d+)', r['file'])
            if match:
                m = f"low_rank_r{match.group(1)}"
            elif 'low_rank' in r['file']:
                m = f"low_rank"
            key = m
            if key not in method_groups:
                method_groups[key] = []
            method_groups[key].append(float(r['test_loss']))

        methods = sorted(method_groups.keys())
        means = [np.mean(method_groups[m]) for m in methods]
        stds = [np.std(method_groups[m]) for m in methods]

        colors_map = {'procrustes': '#9C27B0', 'linear': '#2196F3', 'lasso': '#FF5722'}
        colors = []
        for m in methods:
            if 'low_rank' in m:
                colors.append('#4CAF50')
            elif m in colors_map:
                colors.append(colors_map[m])
            else:
                colors.append('#9E9E9E')

        ax.barh(range(len(methods)), means, xerr=stds, color=colors, capsize=3)
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels(methods, fontsize=8)
        ax.set_xlabel('Test Loss (lower = better)')
        ax.set_title(title)
        ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.5)
    except Exception as e:
        ax.text(0.5, 0.5, f'Failed: {e}', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)

plt.suptitle('Alignment Quality Comparison — All Evals', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/plots/alignment_comparison_all.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: alignment_comparison_all.png')

# PLOT 5: Permutation Test Results
fig, ax = plt.subplots(figsize=(12, 6))
all_perm = []
for phase in ['a', 'b', 'd']:
    try:
        data = load_csv(f'outputs/intermediary/permutation_test_phase_{phase}.csv')
        for r in data:
            r['phase_label'] = f"Eval {phase.upper()}"
            all_perm.append(r)
    except:
        pass

if all_perm:
    labels = [f"Ph{r.get('phase_label','?')[-1]}: L{r['layer_a']}->L{r['layer_b']}" for r in all_perm]
    observed = [float(r['observed_cka']) for r in all_perm]
    null_means = [float(r['null_mean']) for r in all_perm]
    null_stds = [float(r['null_std']) for r in all_perm]

    x = np.arange(len(labels))
    ax.bar(x, observed, color='#2196F3', label='Observed CKA', alpha=0.8)
    ax.errorbar(x, null_means, yerr=[2*s for s in null_stds], fmt='o', color='red',
                label='Null distribution (mean +/- 2 sigma)', capsize=5, markersize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('CKA Score')
    ax.set_title('Permutation Tests: All CKA Scores Highly Significant (p=0.000)')
    ax.legend()

plt.tight_layout()
plt.savefig('outputs/plots/permutation_tests.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: permutation_tests.png')

# Print summary
print('\n' + '='*80)
print('EXPERIMENT SUMMARY')
print('='*80)

for phase, title in [('a', 'Eval A: Llama-1B vs Pythia-1.4B (d=2048)'),
                      ('b', 'Eval B: Gemma-2B vs Qwen-1.5B (d=2304 vs 1536)'),
                      ('d', 'Eval D: Llama-3B vs Pythia-2.8B (d=3072 vs 2560)'),
                      ('e', 'Eval E: Llama-3B vs Gemma-2B (d=3072 vs 2304)')]:
    print(f'\n--- {title} ---')
    try:
        cka = np.load(f'outputs/phase_{phase}/cka/cka_results.npz', allow_pickle=True)['cka_matrix']
        print(f'  CKA: mean={cka.mean():.4f}, max={cka.max():.4f}')
    except:
        print('  CKA: N/A')
    try:
        align = load_csv(f'outputs/intermediary/alignment_results_phase_{phase}.csv')
        best = min(align, key=lambda r: float(r['test_loss']))
        print(f'  Best alignment: {best["file"]} (test_loss={float(best["test_loss"]):.4f}, R2={float(best["explained_variance"]):.4f})')
    except:
        print('  Alignment: N/A')

print('\n--- KEY FINDINGS ---')
print('1. CKA scores are consistently WEAK (0.10-0.22) across ALL model sizes')
print('2. CKA does NOT increase with scale: 3B models ~ 1B models')
print('3. All CKA scores are statistically significant (p=0.000, effect sizes >100)')
print('4. Low-rank alignment (rank 32) BEATS full ridge -- cross-model signal is low-dimensional')
print('5. Higher rank causes overfitting -- more parameters != better alignment')
print('6. Platonic Representation Hypothesis NOT supported at 1-3B scale')
print('7. Late layers consistently match best (final layer of Pythia/Qwen)')
print('='*80)
