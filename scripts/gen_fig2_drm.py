"""
Generate Figure 2: DRM False Memory Geometry
=============================================
4-panel figure showing DRM false memory phenomena in HIDE embedding space.
  (a) UMAP of SLEEP list with lure geometry
  (b) False alarm rates comparison (HIDE vs human)
  (c) Threshold operating curve
  (d) Per-list lure similarity dot plot
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy.spatial import ConvexHull

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "paper"))

from figure_style import (
    set_nature_style, COLORS, MARKERS, FULL_WIDTH,
    panel_label, human_reference_line, save_figure,
)

# ─────────────────────────────────────────────────────
# DRM Word Lists (from experiments/phase5/run_phase5.py)
# ─────────────────────────────────────────────────────

DRM_LISTS = {
    "SLEEP": {"studied": ["bed","rest","awake","tired","dream","wake","snooze","blanket","doze","slumber","snore","nap","peace","yawn","drowsy"], "lure": "sleep"},
    "NEEDLE": {"studied": ["thread","pin","eye","sewing","sharp","point","prick","thimble","haystack","thorn","hurt","injection","syringe","cloth","knitting"], "lure": "needle"},
    "ROUGH": {"studied": ["smooth","bumpy","road","tough","sandpaper","jagged","ready","coarse","uneven","riders","rugged","sand","boards","ground","gravel"], "lure": "rough"},
    "SWEET": {"studied": ["sour","candy","sugar","bitter","good","taste","tooth","nice","honey","soda","chocolate","heart","cake","tart","pie"], "lure": "sweet"},
    "CHAIR": {"studied": ["table","sit","legs","seat","couch","desk","recliner","sofa","wood","cushion","swivel","stool","sitting","rocking","bench"], "lure": "chair"},
    "WINDOW": {"studied": ["door","glass","pane","shade","ledge","sill","house","open","curtain","frame","view","breeze","sash","screen","shutter"], "lure": "window"},
    "SMELL": {"studied": ["nose","breathe","sniff","aroma","hear","see","nostril","whiff","scent","reek","stink","fragrance","perfume","salts","rose"], "lure": "smell"},
    "MOUNTAIN": {"studied": ["hill","valley","climb","summit","top","molehill","peak","plain","glacier","goat","bike","climber","range","steep","ski"], "lure": "mountain"},
    "MUSIC": {"studied": ["note","sound","piano","sing","radio","band","melody","horn","concert","instrument","symphony","jazz","orchestra","art","rhythm"], "lure": "music"},
    "COLD": {"studied": ["hot","snow","warm","winter","ice","wet","frigid","chilly","heat","weather","freeze","air","shiver","arctic","frost"], "lure": "cold"},
    "ANGER": {"studied": ["mad","fear","hate","rage","temper","fury","ire","wrath","happy","fight","hatred","mean","calm","emotion","enrage"], "lure": "anger"},
    "DOCTOR": {"studied": ["nurse","sick","lawyer","medicine","health","hospital","dentist","physician","ill","patient","office","stethoscope","surgeon","clinic","cure"], "lure": "doctor"},
    "RIVER": {"studied": ["water","stream","lake","Mississippi","boat","tide","swim","flow","run","barge","creek","brook","fish","bridge","winding"], "lure": "river"},
    "FRUIT": {"studied": ["apple","vegetable","orange","kiwi","citrus","ripe","pear","banana","berry","cherry","basket","juice","salad","bowl","cocktail"], "lure": "fruit"},
    "BLACK": {"studied": ["white","dark","cat","charcoal","night","funeral","color","grief","blue","death","ink","bottom","coal","brown","gray"], "lure": "black"},
    "KING": {"studied": ["queen","England","crown","prince","George","dictator","palace","throne","chess","rule","subjects","monarch","royal","leader","reign"], "lure": "king"},
    "BREAD": {"studied": ["butter","food","eat","sandwich","rye","jam","milk","flour","jelly","dough","crust","slice","wine","loaf","toast"], "lure": "bread"},
    "SPIDER": {"studied": ["web","insect","bug","fright","fly","arachnid","crawl","tarantula","poison","bite","creepy","animal","ugly","feelers","small"], "lure": "spider"},
    "SLOW": {"studied": ["fast","lethargic","stop","listless","snail","cautious","delay","traffic","turtle","hesitant","speed","quick","sluggish","wait","molasses"], "lure": "slow"},
    "MAN": {"studied": ["woman","husband","uncle","lady","mouse","male","father","strong","friend","beard","person","handsome","muscle","suit","old"], "lure": "man"},
    "SOFT": {"studied": ["hard","light","pillow","plush","loud","cotton","fur","touch","fluffy","feather","tender","skin","silk","smooth","kitten"], "lure": "soft"},
    "THIEF": {"studied": ["steal","robber","crook","burglar","money","cop","bad","rob","jail","gun","villain","crime","bank","bandit","criminal"], "lure": "thief"},
    "HIGH": {"studied": ["low","clouds","up","tall","tower","jump","above","building","noon","over","airplane","dive","elevate","cliff","sky"], "lure": "high"},
    "LION": {"studied": ["tiger","circus","jungle","tamer","den","cub","Africa","mane","cage","feline","roar","fierce","wildcat","pride","cougar"], "lure": "lion"},
}

UNRELATED_WORDS = ["bicycle", "computer", "elephant", "volcano", "umbrella"]

SEEDS = [42, 123, 456, 789, 1024]

# Human reference values (Roediger & McDermott 1995)
HUMAN_HIT_RATE = 0.72
HUMAN_LURE_FA = 0.55
HUMAN_UNRELATED_FA = 0.16


def load_results():
    """Load DRM results from all 5 seeds."""
    results = []
    results_dir = PROJECT_ROOT / "results" / "phase5"
    for seed in SEEDS:
        path = results_dir / f"results_seed{seed}.json"
        with open(path) as f:
            data = json.load(f)
        results.append(data["drm"])
    return results


def encode_words_bge(words):
    """Encode words with bge-large-en-v1.5 on cuda:1."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cuda:1")
    embeddings = model.encode(words, normalize_embeddings=True, show_progress_bar=False)
    return embeddings


def panel_a_umap(ax, embeddings_studied, embeddings_lure, embeddings_unrelated,
                 studied_words, lure_word, unrelated_words):
    """Panel a: 2D UMAP of SLEEP list with lure geometry."""
    import umap

    # Combine all embeddings
    all_embs = np.vstack([embeddings_studied, embeddings_lure, embeddings_unrelated])
    all_labels = studied_words + [lure_word] + unrelated_words
    n_studied = len(studied_words)
    n_lure = 1
    n_unrelated = len(unrelated_words)

    # UMAP projection
    reducer = umap.UMAP(
        n_neighbors=15, min_dist=0.1, n_components=2,
        random_state=42, metric="cosine",
    )
    coords = reducer.fit_transform(all_embs)

    studied_coords = coords[:n_studied]
    lure_coords = coords[n_studied:n_studied + n_lure]
    unrelated_coords = coords[n_studied + n_lure:]

    # Draw convex hull around studied words as shaded region
    if len(studied_coords) >= 3:
        hull = ConvexHull(studied_coords)
        hull_pts = studied_coords[hull.vertices]
        # Close the polygon
        hull_pts = np.vstack([hull_pts, hull_pts[0]])
        ax.fill(hull_pts[:, 0], hull_pts[:, 1],
                alpha=0.12, color=COLORS['primary'], zorder=0)
        ax.plot(hull_pts[:, 0], hull_pts[:, 1],
                color=COLORS['primary'], alpha=0.3, linewidth=0.8, linestyle='--', zorder=1)

    # Plot studied words
    ax.scatter(studied_coords[:, 0], studied_coords[:, 1],
               c=COLORS['primary'], s=35, alpha=0.85, zorder=3,
               edgecolors='white', linewidths=0.3, label='Studied words')

    # Plot critical lure
    ax.scatter(lure_coords[:, 0], lure_coords[:, 1],
               c=COLORS['secondary'], s=150, marker='*', zorder=5,
               edgecolors='white', linewidths=0.5, label='Critical lure')

    # Plot unrelated words
    ax.scatter(unrelated_coords[:, 0], unrelated_coords[:, 1],
               c=COLORS['neutral'], s=28, alpha=0.7, zorder=3,
               edgecolors='white', linewidths=0.3, label='Unrelated words')

    # Add word labels using adjustText to avoid overlap
    from adjustText import adjust_text
    fontsize_label = 5.5
    texts_studied = []
    for i, word in enumerate(studied_words):
        t = ax.text(studied_coords[i, 0], studied_coords[i, 1], word,
                    fontsize=fontsize_label, color=COLORS['primary'], alpha=0.85)
        texts_studied.append(t)

    lure_text = ax.text(lure_coords[0, 0], lure_coords[0, 1], lure_word.upper(),
                         fontsize=7, color=COLORS['secondary'], fontweight='bold')

    texts_unrel = []
    for i, word in enumerate(unrelated_words):
        t = ax.text(unrelated_coords[i, 0], unrelated_coords[i, 1], word,
                    fontsize=fontsize_label, color=COLORS['neutral'], alpha=0.7)
        texts_unrel.append(t)

    # Adjust all labels to avoid overlap
    adjust_text(texts_studied + [lure_text] + texts_unrel, ax=ax)

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("SLEEP list embedding geometry", fontsize=9)
    ax.legend(loc='lower right', fontsize=6, markerscale=0.7,
              handletextpad=0.3, borderpad=0.3)


def panel_b_bars(ax, all_results):
    """Panel b: False alarm rates comparison with human reference."""
    # Collect best_match values across seeds
    studied_vals = []
    lure_vals = []
    unrel_vals = []

    for res in all_results:
        bm = res["best_match"]
        studied_vals.append(bm["hit_rate"])
        lure_vals.append(bm["false_alarm_critical"])
        unrel_vals.append(bm["false_alarm_unrelated"])

    means = [np.mean(studied_vals), np.mean(lure_vals), np.mean(unrel_vals)]
    stds = [np.std(studied_vals), np.std(lure_vals), np.std(unrel_vals)]

    bar_colors = [COLORS['primary'], COLORS['secondary'], COLORS['neutral']]
    labels = ['Studied\nhits', 'Critical\nlure FA', 'Unrelated\nFA']
    x = np.arange(len(labels))
    bar_width = 0.55

    bars = ax.bar(x, means, bar_width, color=bar_colors, alpha=0.85,
                  edgecolor='white', linewidth=0.5, zorder=3)

    # Error bars
    ax.errorbar(x, means, yerr=stds, fmt='none', ecolor='black',
                elinewidth=0.8, capsize=3, capthick=0.8, zorder=4)

    # Human reference diamonds
    human_vals = [HUMAN_HIT_RATE, HUMAN_LURE_FA, HUMAN_UNRELATED_FA]
    ax.scatter(x, human_vals, marker=MARKERS['human'], s=40,
               color=COLORS['human'], zorder=5, edgecolors='white',
               linewidths=0.5, label='Human')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.15)
    ax.set_title(r"Recognition rates ($\theta$=0.82)", fontsize=9)
    ax.legend(loc='upper right', fontsize=6, handletextpad=0.3, borderpad=0.3)

    # Add HIDE value annotations above bars (offset to avoid human diamonds)
    for i, (m, s, hv) in enumerate(zip(means, stds, human_vals)):
        # Place text above whichever is higher: bar+errbar or human diamond
        top_y = max(m + s, hv) + 0.05
        ax.text(i, top_y, f"{m:.2f}", ha='center', va='bottom',
                fontsize=6, color=bar_colors[i], fontweight='bold')


def panel_c_threshold(ax, all_results):
    """Panel c: Threshold operating curve with CI."""
    # Collect threshold sweep data across seeds
    # Each seed has threshold_sweep list
    thresholds = None
    hit_rates_all = []
    fa_crit_all = []
    fa_unrel_all = []

    for res in all_results:
        sweep = res["threshold_sweep"]
        if thresholds is None:
            thresholds = np.array([s["threshold"] for s in sweep])
        hit_rates_all.append([s["hit_rate"] for s in sweep])
        fa_crit_all.append([s["false_alarm_critical"] for s in sweep])
        fa_unrel_all.append([s["false_alarm_unrelated"] for s in sweep])

    hit_rates_all = np.array(hit_rates_all)
    fa_crit_all = np.array(fa_crit_all)
    fa_unrel_all = np.array(fa_unrel_all)

    hit_mean = np.mean(hit_rates_all, axis=0)
    hit_std = np.std(hit_rates_all, axis=0)
    fa_crit_mean = np.mean(fa_crit_all, axis=0)
    fa_crit_std = np.std(fa_crit_all, axis=0)
    fa_unrel_mean = np.mean(fa_unrel_all, axis=0)
    fa_unrel_std = np.std(fa_unrel_all, axis=0)

    # 95% CI = 1.96 * std / sqrt(n)
    n_seeds = len(all_results)
    ci_factor = 1.96 / np.sqrt(n_seeds)

    # Plot lines with shaded CI
    ax.plot(thresholds, hit_mean, color=COLORS['primary'], linewidth=1.2, label='Hit rate')
    ax.fill_between(thresholds,
                    hit_mean - hit_std * ci_factor,
                    hit_mean + hit_std * ci_factor,
                    color=COLORS['primary'], alpha=0.15)

    ax.plot(thresholds, fa_crit_mean, color=COLORS['secondary'], linewidth=1.2, label='Lure FA')
    ax.fill_between(thresholds,
                    fa_crit_mean - fa_crit_std * ci_factor,
                    fa_crit_mean + fa_crit_std * ci_factor,
                    color=COLORS['secondary'], alpha=0.15)

    ax.plot(thresholds, fa_unrel_mean, color=COLORS['neutral'], linewidth=1.2, label='Unrelated FA')
    ax.fill_between(thresholds,
                    fa_unrel_mean - fa_unrel_std * ci_factor,
                    fa_unrel_mean + fa_unrel_std * ci_factor,
                    color=COLORS['neutral'], alpha=0.15)

    # Vertical dashed line at theta=0.82
    ax.axvline(x=0.82, color='black', linestyle=':', linewidth=0.8, alpha=0.6)
    ax.text(0.825, 0.5, r'$\theta$=0.82', fontsize=6, rotation=90,
            va='center', ha='left', alpha=0.7)

    ax.set_xlabel(r"Threshold $\theta$")
    ax.set_ylabel("Rate")
    ax.set_xlim(0.50, 0.95)
    ax.set_ylim(-0.05, 1.08)
    ax.set_title("Threshold operating curve", fontsize=9)
    ax.legend(loc='center left', fontsize=6, handletextpad=0.3, borderpad=0.3)


def panel_d_perlist(ax, all_results):
    """Panel d: Per-list lure similarity dot plot sorted by value."""
    # Aggregate lure_sim per list across seeds
    list_names = [item["list_name"] for item in all_results[0]["per_list"]]
    n_lists = len(list_names)

    lure_sims_by_list = {name: [] for name in list_names}
    for res in all_results:
        for item in res["per_list"]:
            lure_sims_by_list[item["list_name"]].append(item["lure_sim"])

    # Compute means
    mean_sims = {name: np.mean(vals) for name, vals in lure_sims_by_list.items()}
    std_sims = {name: np.std(vals) for name, vals in lure_sims_by_list.items()}

    # Sort by mean similarity
    sorted_names = sorted(mean_sims.keys(), key=lambda n: mean_sims[n])
    sorted_means = [mean_sims[n] for n in sorted_names]
    sorted_stds = [std_sims[n] for n in sorted_names]

    y_pos = np.arange(n_lists)

    # Horizontal dot plot with error bars
    ax.errorbar(sorted_means, y_pos, xerr=sorted_stds, fmt='o',
                color=COLORS['primary'], markersize=4, ecolor=COLORS['light_neutral'],
                elinewidth=0.6, capsize=0, zorder=3)

    # Vertical dashed line at overall mean
    overall_mean = np.mean(sorted_means)
    ax.axvline(x=overall_mean, color=COLORS['secondary'], linestyle='--',
               linewidth=0.8, alpha=0.7)
    ax.text(overall_mean + 0.003, n_lists - 0.5,
            f"mean={overall_mean:.3f}", fontsize=5.5,
            color=COLORS['secondary'], va='top', ha='left', fontstyle='italic')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_names, fontsize=4.5, fontfamily='monospace')
    ax.set_xlabel("Lure cosine similarity")
    ax.set_title("Per-list lure similarity", fontsize=9)
    # Set xlim based on data range with padding
    xmin = min(sorted_means) - max(sorted_stds) - 0.02
    xmax = max(sorted_means) + max(sorted_stds) + 0.02
    ax.set_xlim(max(0.65, xmin), min(0.98, xmax))
    ax.set_ylim(-0.5, n_lists - 0.5)


def main():
    set_nature_style()

    print("Loading DRM results from 5 seeds...")
    all_results = load_results()

    # ── Encode words for UMAP panel ──
    print("Encoding SLEEP list words with bge-large-en-v1.5 on cuda:1...")
    sleep_data = DRM_LISTS["SLEEP"]
    studied_words = sleep_data["studied"]
    lure_word = sleep_data["lure"]

    all_words = studied_words + [lure_word] + UNRELATED_WORDS
    all_embeddings = encode_words_bge(all_words)

    emb_studied = all_embeddings[:len(studied_words)]
    emb_lure = all_embeddings[len(studied_words):len(studied_words) + 1]
    emb_unrelated = all_embeddings[len(studied_words) + 1:]

    # ── Create figure ──
    print("Generating figure...")
    fig = plt.figure(figsize=(FULL_WIDTH, FULL_WIDTH * 0.78))

    # Gridspec: left 50% for panel a, right 50% split into 3 rows for b/c/d
    # Give panel d extra height for the 24-list dot plot
    gs = gridspec.GridSpec(
        3, 2, figure=fig,
        width_ratios=[1, 1],
        height_ratios=[0.85, 0.85, 1.5],
        hspace=0.52, wspace=0.35,
    )

    ax_a = fig.add_subplot(gs[:, 0])       # Left half, all rows
    ax_b = fig.add_subplot(gs[0, 1])       # Top right
    ax_c = fig.add_subplot(gs[1, 1])       # Middle right
    ax_d = fig.add_subplot(gs[2, 1])       # Bottom right

    # ── Panel a: UMAP ──
    panel_a_umap(ax_a, emb_studied, emb_lure, emb_unrelated,
                 studied_words, lure_word, UNRELATED_WORDS)
    panel_label(ax_a, 'a', x=-0.08, y=1.03)

    # ── Panel b: Bar chart ──
    panel_b_bars(ax_b, all_results)
    panel_label(ax_b, 'b', x=-0.18, y=1.10)

    # ── Panel c: Threshold curve ──
    panel_c_threshold(ax_c, all_results)
    panel_label(ax_c, 'c', x=-0.18, y=1.10)

    # ── Panel d: Per-list dot plot ──
    panel_d_perlist(ax_d, all_results)
    panel_label(ax_d, 'd', x=-0.18, y=1.10)

    # ── Save ──
    save_figure(fig, "fig2_drm")
    print("Done.")


if __name__ == "__main__":
    main()
