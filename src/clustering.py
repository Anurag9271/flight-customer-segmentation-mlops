import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage


# ── PCA ──────────────────────────────────────────────────────────────────────

def apply_pca(X, n_components=2):
    """
    Reduce dimensions using PCA.
    We use 2 components so we can visualise clusters on a 2D scatter plot.
    Clustering itself runs on the full scaled LRFMC — PCA is only for plotting.
    """
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)

    explained = pca.explained_variance_ratio_ * 100
    print(f"PC1: {explained[0]:.1f}%  |  PC2: {explained[1]:.1f}%  |  Total: {sum(explained):.1f}% variance kept")

    return X_pca, pca


# ── ELBOW METHOD ─────────────────────────────────────────────────────────────

def compute_elbow(X, k_range=range(2, 11)):
    """
    Compute inertia for each K to draw the elbow curve.
    Inertia = total distance of every point from its cluster centre.
    Lower inertia is better — but it always drops as K increases,
    so we look for the 'elbow' where the drop suddenly slows down.
    """
    wcss = []
    for k in k_range:
        model = KMeans(n_clusters=k, init='k-means++',
                       n_init=10, random_state=42)
        model.fit(X)
        wcss.append(model.inertia_)
        print(f"  K={k}  Inertia: {model.inertia_:,.0f}")
    return wcss


def plot_elbow(wcss, k_range=range(2, 11)):
    """Plot the elbow curve to visually find the best K."""
    plt.figure(figsize=(9, 5))
    plt.plot(list(k_range), wcss, 'o-', color='#185FA5',
             linewidth=2.5, markersize=8)
    plt.xlabel('Number of Clusters (K)', fontsize=12)
    plt.ylabel('Inertia (WCSS)', fontsize=12)
    plt.title('Elbow Method — Finding Optimal K', fontsize=13, fontweight='bold')
    plt.xticks(list(k_range))
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('../outputs/reports/elbow_curve.png', dpi=150, bbox_inches='tight')
    plt.show()


# ── SILHOUETTE SCORE ─────────────────────────────────────────────────────────

def plot_silhouette_scores(X, k_range=range(2, 11)):
    """
    Compute and plot silhouette score for each K.
    Silhouette measures how well each customer fits its own cluster
    vs how poorly it fits the nearest other cluster.
    Range: -1 to 1. Higher is better. Peak = best K.
    """
    scores = []
    for k in k_range:
        labels = KMeans(n_clusters=k, init='k-means++',
                        n_init=10, random_state=42).fit_predict(X)
        score = silhouette_score(X, labels, sample_size=10000, random_state=42)
        scores.append(score)
        print(f"  K={k}  Silhouette Score: {score:.4f}")

    best_k = list(k_range)[scores.index(max(scores))]

    plt.figure(figsize=(9, 5))
    plt.plot(list(k_range), scores, 's-', color='#0F6E56',
             linewidth=2.5, markersize=8)
    plt.axvline(x=best_k, color='#D85A30', linestyle='--',
                linewidth=1.8, label=f'Best K = {best_k}  (score = {max(scores):.4f})')
    plt.xlabel('Number of Clusters (K)', fontsize=12)
    plt.ylabel('Silhouette Score', fontsize=12)
    plt.title('Silhouette Score by K', fontsize=13, fontweight='bold')
    plt.xticks(list(k_range))
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('../outputs/reports/silhouette_scores.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\n  Best K by silhouette: {best_k}")
    return scores, best_k


# ── K-MEANS ───────────────────────────────────────────────────────────────────

def run_kmeans(X, n_clusters=4):
    """
    Fit K-Means with the chosen optimal K.
    init='k-means++' chooses smarter starting centroids — avoids
    bad random initialisation that leads to poor clustering.
    n_init=10 runs it 10 times and keeps the best result.
    """
    model = KMeans(n_clusters=n_clusters, init='k-means++',
                   n_init=10, random_state=42)
    labels = model.fit_predict(X)
    print(f"K-Means fitted with K={n_clusters}")
    return model, labels


# ── AGGLOMERATIVE HIERARCHICAL ────────────────────────────────────────────────
#
# def plot_dendrogram(X, sample_size=2000):
#     """
#     Plot a dendrogram on a sample of the data.
#     Shows how customers merge into groups as distance threshold increases.
#     The height of each merge = how different the groups are.
#     Look for the largest vertical gap — cut there to get natural clusters.
#     We sample because computing linkage on 62K rows is slow.
#     """
#     np.random.seed(42)
#     idx = np.random.choice(len(X), size=min(sample_size, len(X)), replace=False)
#     sample = X[idx] if isinstance(X, np.ndarray) else X.iloc[idx].values
#
#     print(f"Computing Ward linkage on {len(sample):,} sample rows...")
#     Z = linkage(sample, method='ward')
#
#     plt.figure(figsize=(14, 6))
#     dendrogram(Z, truncate_mode='lastp', p=30,
#                leaf_rotation=90, leaf_font_size=9,
#                show_contracted=True,
#                color_threshold=0.7 * max(Z[:, 2]))
#     plt.title('Hierarchical Clustering Dendrogram (Ward Linkage)',
#               fontsize=13, fontweight='bold')
#     plt.xlabel('Cluster / Sample Index')
#     plt.ylabel('Ward Distance')
#     # plt.axhline(y=0.7 * max(Z[:, 2]), color='#D85A30',
#     #             linestyle='--', alpha=0.7, label='Suggested cut')
#     plt.legend()
#     plt.grid(axis='y', alpha=0.3)
#     plt.tight_layout()
#     plt.savefig('../outputs/reports/dendrogram.png', dpi=150, bbox_inches='tight')
#     plt.show()


def run_hierarchical(X, n_clusters=4, linkage='ward', sample_size=10000):
    """
    Fit Agglomerative Hierarchical Clustering on a sample.

    WHY SAMPLE? Agglomerative computes distances between every pair
    of points. With 62K rows that needs ~15GB RAM — impossible on
    a normal machine. A 10,000 row sample gives reliable cluster
    labels for comparison purposes.

    Ward linkage minimises within-cluster variance — same goal as
    K-Means, so the comparison between both is meaningful.
    """
    np.random.seed(42)
    idx = np.random.choice(len(X), size=min(sample_size, len(X)), replace=False)
    X_sample = X[idx] if isinstance(X, np.ndarray) else X.iloc[idx].values

    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    sample_labels = model.fit_predict(X_sample)

    print(f"Hierarchical Clustering fitted on {len(X_sample):,} sample rows")
    print(f"K={n_clusters}, linkage='{linkage}'")

    return model, sample_labels, idx


# ── DBSCAN ────────────────────────────────────────────────────────────────────

def run_dbscan(X, eps=0.8, min_samples=10):
    """
    Fit DBSCAN — density-based clustering.
    Does NOT need K specified — finds clusters by density automatically.
    eps       = max distance between two points to be considered neighbours
    min_samples = minimum points to form a dense region (core point)
    Label -1  = noise / outlier — customer doesn't fit any cluster
    These noise points are often your ultra-VIP outlier customers.
    """
    model = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    labels = model.fit_predict(X)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    print(f"DBSCAN found {n_clusters} clusters  |  Noise points: {n_noise:,} ({n_noise / len(labels) * 100:.1f}%)")

    return model, labels


# ── CLUSTER VISUALISATION IN PCA SPACE ───────────────────────────────────────

def plot_clusters_pca(X_pca, labels, title='Cluster Plot', save_name='clusters'):
    """
    Scatter plot of clusters in 2D PCA space.
    Each colour = one cluster. Lets you visually judge how
    well separated the clusters are from each other.
    """
    unique_labels = sorted(set(labels))
    colors = ['#185FA5', '#0F6E56', '#D85A30', '#993556',
              '#534AB7', '#854F0B', '#A32D2D']

    plt.figure(figsize=(9, 6))
    for i, label in enumerate(unique_labels):
        mask = labels == label
        color = '#AAAAAA' if label == -1 else colors[i % len(colors)]
        name = 'Noise' if label == -1 else f'Cluster {label}'
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                    c=color, label=name, alpha=0.35, s=8, linewidths=0)

    plt.xlabel('PC1', fontsize=11)
    plt.ylabel('PC2', fontsize=11)
    plt.title(title, fontsize=13, fontweight='bold')
    plt.legend(markerscale=4, fontsize=9)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(f'../outputs/reports/{save_name}.png', dpi=150, bbox_inches='tight')
    plt.show()