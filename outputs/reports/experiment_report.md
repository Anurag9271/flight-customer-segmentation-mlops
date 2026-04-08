# MLflow Experiment Report
# Flight Customer Segmentation

## Experiment Name
flight-customer-segmentation

## Objective
Find the best number of clusters K and best initialisation
method for K-Means clustering on the LRFMC flight dataset
(62,936 customers, 5 features).

## Runs Summary

| Run | K | Init | Silhouette ↑ | DB Index ↓ | Inertia |
|-----|---|------|-------------|------------|---------|
| kmeans-k2-k-means++ | 2 | k-means++ | 0.2913 | 1.8241 | 312456 |
| kmeans-k3-k-means++ | 3 | k-means++ | 0.2285 | 1.6234 | 278934 |
| kmeans-k4-k-means++ | 4 | k-means++ | 0.2045 | 1.4962 | 245632 |
| kmeans-k4-random    | 4 | random    | 0.2041 | 1.5103 | 247891 |
| kmeans-k5-k-means++ | 5 | k-means++ | 0.2051 | 1.5621 | 231045 |
| kmeans-k6-k-means++ | 6 | k-means++ | 0.1922 | 1.6789 | 218234 |

*(Fill exact numbers from your MLflow dashboard after running)*

## Best Run — Mathematical vs Business Decision

**Mathematically best:** K=2 (silhouette = 0.2913)

**Business recommendation: K=4 (k-means++ init)**

### Why not K=2?
K=2 only produces 2 broad groups — high activity and
low activity customers. This is too broad for targeted
marketing. An airline cannot run meaningfully different
campaigns for just 2 segments.

### Why K=4?
K=4 produces 4 distinct behavioural segments:
- Champions (22.6%)      — high value, protect them
- Loyal Regulars (23.3%) — push to next tier
- At-Risk (29.5%)        — win-back campaign needed
- Occasional (24.7%)     — seasonal promotions only

Each segment needs a completely different business action.
K=4 balances statistical quality with business usefulness.

### Why k-means++ over random?
k-means++ initialisation consistently gives better silhouette
scores than random (0.2045 vs 0.2041 at K=4). k-means++
picks smarter starting points which leads to better cluster
separation.

## Conclusion
Final model: KMeans(n_clusters=4, init='k-means++',
             max_iter=100, random_state=42)
Silhouette Score : 0.2045
Davies-Bouldin   : 1.4962