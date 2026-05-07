# cluster_content.py

Reads the flat CSV produced by `scrape_subreddit_content.py` and groups posts and comments into thematic clusters, then saves an interactive visualisation and an annotated CSV.

## Pipeline

```
flat CSV  →  sentence embeddings  →  UMAP 2D  →  HDBSCAN clusters  →  TF-IDF labels
         →  results/{sub}_clusters.html  (interactive scatter plot)
         →  results/{sub}_clustered.csv  (original rows + cluster columns)
```

### 1. Load & filter
Rows with deleted (`[deleted]`, `[removed]`) or very short text are dropped. You can optionally restrict the analysis to posts only, comments only, or both.

### 2. Sentence embeddings
Each piece of text is converted into a dense vector (384 numbers by default) using a **sentence-transformer** model (`all-MiniLM-L6-v2`). Two texts that mean similar things will have vectors that are close together in this space, regardless of exact wording. The model is downloaded automatically on first run and cached locally.

For subreddits with significant German content, swap the model for `paraphrase-multilingual-MiniLM-L12-v2` via `--model`.

### 3. UMAP dimensionality reduction
384 dimensions cannot be plotted or clustered efficiently. **UMAP** compresses the vectors to 2D while preserving neighbourhood structure — items that were close in the full embedding space stay close in the 2D projection. The result is a 2D coordinate for every post/comment.

### 4. HDBSCAN clustering
**HDBSCAN** finds dense regions in the 2D space and labels them as clusters. Unlike k-means it does not require you to specify the number of clusters in advance, and it explicitly marks outliers as **noise** (cluster ID `-1`) rather than forcing them into the nearest group. The key parameter is `--min-cluster-size`: the minimum number of items that can form a cluster.

### 5. TF-IDF cluster labels
Each cluster is automatically labelled with its 5 most distinctive words, computed by fitting a TF-IDF matrix across all texts and taking the highest-scoring terms in the cluster's centroid vector. These labels appear in the legend of the HTML plot and in the `cluster_label` column of the CSV.

## Outputs

| File | Description |
|---|---|
| `results/{sub}_clusters.html` | Interactive Plotly scatter — hover over any point to see author, date, score, and a text preview. Each cluster is a separate colour; noise points are translucent grey. Open in any browser, no server needed. |
| `results/{sub}_clustered.csv` | The original flat CSV with two extra columns: `cluster_id` (integer, -1 = noise) and `cluster_label` (the TF-IDF keyword string). Use this as input for the LLM classification step. |

## Usage

```bash
# basic — all posts and comments
python cluster_content.py --input results/ethz_flat.csv

# posts only, tighter clusters
python cluster_content.py --input results/ethz_flat.csv --filter-type post --min-cluster-size 8

# multilingual model for German/English mixed content
python cluster_content.py --input results/ethz_flat.csv --model paraphrase-multilingual-MiniLM-L12-v2
```

## Key parameters

| Flag | Default | Effect |
|---|---|---|
| `--min-cluster-size` | 10 | Lower → more, smaller clusters. Raise if you get too many tiny clusters. |
| `--umap-neighbors` | 15 | Lower → captures fine local structure. Raise for smoother global layout. |
| `--umap-min-dist` | 0.05 | Lower → points pack more tightly in the plot. Raise to spread them out. |
| `--filter-type` | all | `post` / `comment` / `all` |
| `--model` | all-MiniLM-L6-v2 | Any model from [sbert.net/docs/pretrained_models.html](https://www.sbert.net/docs/pretrained_models.html) |

## Relation to Volk et al. (2024)

Volk et al. used manual content coding by trained student coders to assign topics (academic / organisational / other) and tonality (positive / neutral / negative) to tweets. This script replaces that manual step with an unsupervised approach: topics emerge from the data rather than being defined in advance. The cluster labels and the HTML visualisation give you an empirical basis for deciding what categories to use in the downstream LLM classification step.
