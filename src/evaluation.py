import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, davies_bouldin_score


# ── INDIVIDUAL METRICS ────────────────────────────────────────────────────────

def get_silhouette(X, labels, model_name="Model"):
    """
    Silhouette Score — measures how tight and separated clusters are.
    Skips noise points (label = -1) from DBSCAN automatically.
    """
    mask = labels != -1
    score = silhouette_score(X[mask], labels[mask],
                             sample_size=10000, random_state=42)
    print(f"  {model_name}  Silhouette Score : {score:.4f}  (higher is better, max 1.0)")
    return score


def get_davies_bouldin(X, labels, model_name="Model"):
    """
    Davies-Bouldin Index — measures average similarity between
    each cluster and its most similar neighbouring cluster.
    Range: 0 to infinity.
    Lower = better (0 = perfect separation).
    Skips noise points (label = -1) from DBSCAN automatically.
    """
    mask = labels != -1
    score = davies_bouldin_score(X[mask], labels[mask])
    print(f"  {model_name}  Davies-Bouldin   : {score:.4f}  (lower is better, min 0.0)")
    return score


# ── COMPARE ALL ALGORITHMS ────────────────────────────────────────────────────

def compare_all_algorithms(X, labels_dict):
    print("=" * 58)
    print("       Algorithm Comparison — Evaluation Metrics")
    print("=" * 58)

    rows = []
    for name, labels in labels_dict.items():
        mask = labels != -1
        n_clusters = len(set(labels[mask]))
        n_noise = int((labels == -1).sum())

        # Need at least 2 clusters to compute metrics
        if n_clusters < 2:
            print(f"\n  {name}")
            print(f"    Clusters found   : {n_clusters}")
            print(f"    Noise points     : {n_noise}")
            print(f"    Silhouette Score : N/A (only {n_clusters} cluster found)")
            print(f"    Davies-Bouldin   : N/A (only {n_clusters} cluster found)")
            rows.append({
                'Algorithm'        : name,
                'Clusters'         : n_clusters,
                'Noise Points'     : n_noise,
                'Silhouette ↑'     : None,
                'Davies-Bouldin ↓' : None
            })
            continue

        sil = silhouette_score(X[mask], labels[mask],
                               sample_size=10000, random_state=42)
        db  = davies_bouldin_score(X[mask], labels[mask])

        print(f"\n  {name}")
        print(f"    Clusters found   : {n_clusters}")
        print(f"    Noise points     : {n_noise}")
        print(f"    Silhouette Score : {sil:.4f}")
        print(f"    Davies-Bouldin   : {db:.4f}")

        rows.append({
            'Algorithm'        : name,
            'Clusters'         : n_clusters,
            'Noise Points'     : n_noise,
            'Silhouette ↑'     : round(sil, 4),
            'Davies-Bouldin ↓' : round(db, 4)
        })

    print("\n" + "=" * 58)
    df = pd.DataFrame(rows).set_index('Algorithm')
    print("\n", df.to_string())
    return df


# def plot_metrics_comparison(comparison_df):
#     fig, axes = plt.subplots(1, 2, figsize=(12, 5))
#     fig.suptitle('Algorithm Comparison — Evaluation Metrics',
#                  fontsize=14, fontweight='bold')
#
#     algos  = comparison_df.index.tolist()
#     colors = ['#185FA5', '#0F6E56', '#D85A30']
#
#     # Silhouette — higher is better
#     sil_vals = comparison_df['Silhouette ↑'].values
#     sil_plot = [v if v is not None else 0 for v in sil_vals]
#     axes[0].bar(algos, sil_plot, color=colors[:len(algos)], edgecolor='white')
#     axes[0].set_title('Silhouette Score (higher = better)', fontsize=11)
#     axes[0].set_ylabel('Score')
#     for i, v in enumerate(sil_vals):
#         label = f'{v:.4f}' if v is not None else 'N/A'
#         axes[0].text(i, (sil_plot[i] or 0) + 0.002, label,
#                      ha='center', fontsize=10)
#     axes[0].grid(axis='y', alpha=0.3)
#
#     # Davies-Bouldin — lower is better
#     db_vals  = comparison_df['Davies-Bouldin ↓'].values
#     db_plot  = [v if v is not None else 0 for v in db_vals]
#     axes[1].bar(algos, db_plot, color=colors[:len(algos)], edgecolor='white')
#     axes[1].set_title('Davies-Bouldin Index (lower = better)', fontsize=11)
#     axes[1].set_ylabel('Score')
#     for i, v in enumerate(db_vals):
#         label = f'{v:.4f}' if v is not None else 'N/A'
#         axes[1].text(i, (db_plot[i] or 0) + 0.01, label,
#                      ha='center', fontsize=10)
#     axes[1].grid(axis='y', alpha=0.3)
#
#     plt.tight_layout()
#     plt.savefig('../outputs/reports/metrics_comparison.png',
#                 dpi=150, bbox_inches='tight')
#     plt.show()
#

# ── CLUSTER PROFILES ─────────────────────────────────────────────────────────

def plot_cluster_profiles(lrfmc_df, labels, algo_name='K-Means'):
    """
    Bar chart showing the mean L, R, F, M, C for each cluster.
    Normalised to 0-1 so all 5 features are on the same scale.
    This is how you 'read' what each cluster actually represents —
    which dimensions are high, which are low for each group.
    """
    df = lrfmc_df.copy()
    df['Cluster'] = labels
    df = df[df['Cluster'] != -1]

    profile = df.groupby('Cluster')[['L', 'R', 'F', 'M', 'C']].mean().round(3)

    # Normalise for visual comparison
    profile_norm = (profile - profile.min()) / (profile.max() - profile.min() + 1e-9)

    ax = profile_norm.plot(kind='bar', figsize=(12, 5),
                           color=['#185FA5', '#D85A30', '#0F6E56', '#993556', '#534AB7'],
                           edgecolor='white', linewidth=0.5)
    ax.set_title(f'{algo_name} — Normalised LRFMC Profile per Cluster',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Normalised Mean (0 = lowest, 1 = highest)')
    ax.set_xticklabels([f'Cluster {i}' for i in profile_norm.index], rotation=0)
    ax.legend(title='LRFMC', bbox_to_anchor=(1.01, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'../outputs/reports/{algo_name.lower().replace(" ","_")}_profiles.png',
                dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\n{algo_name} — Raw Mean LRFMC Values per Cluster:")
    print(profile.to_string())
    return profile


# def plot_feature_boxplots(lrfmc_df, labels, algo_name='K-Means'):
#     """
#     Boxplots for each LRFMC feature split by cluster.
#     Shows the spread of values inside each cluster — not just the mean.
#     Wide boxes = mixed customers. Narrow boxes = very pure cluster.
#     """
#     df = lrfmc_df.copy()
#     df['Cluster'] = labels.astype(str)
#     df = df[df['Cluster'] != '-1']
#
#     fig, axes = plt.subplots(1, 5, figsize=(20, 5))
#     fig.suptitle(f'{algo_name} — LRFMC Feature Distribution per Cluster',
#                  fontsize=13, fontweight='bold')
#
#     colors = ['#E6F1FB', '#FAEEDA', '#EAF3DE', '#FBEAF0', '#EEEDFE']
#     features = ['L', 'R', 'F', 'M', 'C']
#
#     for ax, feat, color in zip(axes, features, colors):
#         order = sorted(df['Cluster'].unique(), key=lambda x: int(x))
#         sns.boxplot(data=df, x='Cluster', y=feat, order=order,
#                     color=color, linewidth=0.9, fliersize=2, ax=ax)
#         ax.set_title(feat, fontsize=12, fontweight='bold')
#         ax.set_xlabel('Cluster')
#         ax.set_ylabel('')
#         ax.grid(axis='y', alpha=0.3)
#
#     plt.tight_layout()
#     plt.savefig(f'../outputs/reports/{algo_name.lower().replace(" ","_")}_boxplots.png',
#                 dpi=150, bbox_inches='tight')
#     plt.show()