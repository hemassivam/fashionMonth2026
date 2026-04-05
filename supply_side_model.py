"""
supply_side_model.py
====================
Supply-Side Trend Detection Pipeline
Uses CLIP embeddings + clustering to identify recurring visual elements
across runway collections, then scores them as candidate trends.

Pipeline:
    1. Load runway images (organised by fashion house)
    2. Extract CLIP embeddings
    3. Cluster embeddings (UMAP + HDBSCAN)
    4. Score clusters by cross-house frequency
    5. Label clusters with CLIP zero-shot classification
    6. Export candidate trends -> candidate_trends.json
"""

import os
import json
import logging
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# ── optional heavy deps (installed separately) ────────────────────────────────
try:
    import clip                    # pip install git+https://github.com/openai/CLIP.git
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    logging.warning("CLIP not installed. Run: pip install git+https://github.com/openai/CLIP.git")

try:
    import umap                    # pip install umap-learn
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

try:
    import hdbscan                 # pip install hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False

from sklearn.cluster import KMeans        # pip install scikit-learn
from sklearn.preprocessing import normalize

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    """Central config — edit these values to match your dataset."""

    # Path where runway images live.
    # Required folder structure: images_root/<HouseName>/<image>.jpg
    images_root: str = "data/runway_images"

    # Where outputs are saved
    output_dir: str = "outputs/supply_side"

    # CLIP model variant
    clip_model: str = "ViT-L/14"   # most accurate; use "ViT-B/32" if RAM is tight

    # ── Dimensionality reduction ──────────────────────────────────────────────
    use_umap: bool = True
    umap_n_components: int = 50
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.0

    # ── Clustering ────────────────────────────────────────────────────────────
    use_hdbscan: bool = True
    hdbscan_min_cluster_size: int = 10
    hdbscan_min_samples: int = 5
    kmeans_k: int = 40             # fallback when HDBSCAN is disabled

    # ── Trend scoring ─────────────────────────────────────────────────────────
    # A cluster must span at least this many distinct fashion houses to count
    min_houses_threshold: int = 3

    # ── Zero-shot label vocabulary ────────────────────────────────────────────
    fashion_elements: list = field(default_factory=lambda: [
        # Silhouettes
        "oversized silhouette", "slim tailored silhouette", "voluminous silhouette",
        "asymmetric hemline", "structured shoulders", "dropped waist",
        # Fabrics / textures
        "sheer organza fabric", "leather or faux leather", "velvet fabric",
        "denim", "knitwear", "satin fabric", "faux fur", "sequins and embellishment",
        # Details / motifs
        "floral print", "animal print", "abstract graphic print",
        "cut-out detail", "ruching and draping", "bow detail",
        "cape overlay", "corseted waist", "utilitarian pockets",
        # Colour stories
        "monochromatic all-black look", "bright primary colours",
        "earth tones and neutrals", "pastel palette",
    ])

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 32
    image_extensions: tuple = (".jpg", ".jpeg", ".png", ".webp")


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RunwayImage:
    path: str
    house: str
    embedding: Optional[np.ndarray] = None
    cluster_id: Optional[int] = None
    top_label: Optional[str] = None
    label_score: Optional[float] = None


@dataclass
class CandidateTrend:
    cluster_id: int
    top_label: str
    label_score: float
    image_count: int
    house_count: int
    houses: list
    cross_house_score: float       # image_count x house_count (simple weighted heuristic)
    sample_image_paths: list


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Image loading
# ─────────────────────────────────────────────────────────────────────────────

def load_images(cfg: Config) -> list:
    """
    Walk images_root/<HouseName>/*.jpg and return a list of RunwayImage objects.
    House name is inferred from the immediate parent directory.
    """
    root = Path(cfg.images_root)
    if not root.exists():
        raise FileNotFoundError(
            f"images_root '{cfg.images_root}' not found.\n"
            "Create the folder and organise images as: images_root/<HouseName>/<image>.jpg"
        )

    records = []
    for house_dir in sorted(root.iterdir()):
        if not house_dir.is_dir():
            continue
        house_name = house_dir.name
        for img_path in sorted(house_dir.iterdir()):
            if img_path.suffix.lower() in cfg.image_extensions:
                records.append(RunwayImage(path=str(img_path), house=house_name))

    logger.info(
        f"Found {len(records)} images across "
        f"{len({r.house for r in records})} houses."
    )
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — CLIP embedding extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_embeddings(records: list, cfg: Config) -> np.ndarray:
    """
    Run images through CLIP vision encoder in batches.
    Returns (N, D) float32 array of L2-normalised embeddings.
    """
    if not CLIP_AVAILABLE:
        raise RuntimeError(
            "CLIP is not installed.\n"
            "Run: pip install git+https://github.com/openai/CLIP.git"
        )

    model, preprocess = clip.load(cfg.clip_model, device=cfg.device)
    model.eval()

    all_embeddings = []

    for i in tqdm(
        range(0, len(records), cfg.batch_size), desc="Extracting CLIP embeddings"
    ):
        batch = records[i : i + cfg.batch_size]
        images, valid_indices = [], []

        for j, rec in enumerate(batch):
            try:
                img = Image.open(rec.path).convert("RGB")
                images.append(preprocess(img))
                valid_indices.append(i + j)
            except Exception as exc:
                logger.warning(f"Skipping {rec.path}: {exc}")

        if not images:
            continue

        tensor = torch.stack(images).to(cfg.device)
        with torch.no_grad():
            feats = model.encode_image(tensor).float().cpu().numpy()

        feats = normalize(feats)   # L2 normalise for cosine similarity
        for k, idx in enumerate(valid_indices):
            records[idx].embedding = feats[k]
        all_embeddings.extend(feats)

    return np.array(all_embeddings, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Dimensionality reduction + clustering
# ─────────────────────────────────────────────────────────────────────────────

def reduce_dimensions(embeddings: np.ndarray, cfg: Config) -> np.ndarray:
    if not cfg.use_umap or not UMAP_AVAILABLE:
        logger.info("Skipping UMAP — clustering raw embeddings.")
        return embeddings

    logger.info(
        f"Running UMAP: {embeddings.shape[1]}D -> {cfg.umap_n_components}D ..."
    )
    reducer = umap.UMAP(
        n_components=cfg.umap_n_components,
        n_neighbors=cfg.umap_n_neighbors,
        min_dist=cfg.umap_min_dist,
        metric="cosine",
        random_state=42,
        verbose=False,
    )
    reduced = reducer.fit_transform(embeddings)
    logger.info("UMAP complete.")
    return reduced


def cluster_embeddings(reduced: np.ndarray, cfg: Config) -> np.ndarray:
    """Returns cluster label array of length N. -1 = noise (HDBSCAN only)."""
    if cfg.use_hdbscan and HDBSCAN_AVAILABLE:
        logger.info("Clustering with HDBSCAN ...")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=cfg.hdbscan_min_cluster_size,
            min_samples=cfg.hdbscan_min_samples,
            metric="euclidean",
            cluster_selection_method="eom",
        )
        labels = clusterer.fit_predict(reduced)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_pct = (labels == -1).mean() * 100
        logger.info(f"HDBSCAN: {n_clusters} clusters | {noise_pct:.1f}% noise points")
    else:
        logger.info(f"Clustering with KMeans (k={cfg.kmeans_k}) ...")
        km = KMeans(n_clusters=cfg.kmeans_k, random_state=42, n_init="auto")
        labels = km.fit_predict(reduced)
        logger.info("KMeans complete.")

    return labels


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Zero-shot CLIP labelling per cluster
# ─────────────────────────────────────────────────────────────────────────────

def label_clusters(
    records: list,
    labels: np.ndarray,
    cfg: Config,
) -> dict:
    """
    For each cluster, compute the mean image embedding, then zero-shot classify
    it against cfg.fashion_elements using CLIP text encodings.
    Returns {cluster_id: (best_label, cosine_score)}.
    """
    if not CLIP_AVAILABLE:
        raise RuntimeError("CLIP required for zero-shot labelling.")

    model, _ = clip.load(cfg.clip_model, device=cfg.device)
    model.eval()

    # Encode text labels once
    prompts = [
        f"a runway fashion look featuring {e}" for e in cfg.fashion_elements
    ]
    tokens = clip.tokenize(prompts).to(cfg.device)
    with torch.no_grad():
        text_feats = model.encode_text(tokens).float()
        text_feats = (
            torch.nn.functional.normalize(text_feats, dim=-1).cpu().numpy()
        )

    # Group embeddings by cluster
    cluster_emb_map = defaultdict(list)
    for rec, lbl in zip(records, labels):
        if rec.embedding is not None and lbl != -1:
            cluster_emb_map[int(lbl)].append(rec.embedding)

    cluster_labels = {}
    for cid, embs in cluster_emb_map.items():
        mean_emb = normalize(np.mean(embs, axis=0, keepdims=True))[0]
        scores = text_feats @ mean_emb     # cosine similarities
        best_idx = int(np.argmax(scores))
        cluster_labels[cid] = (
            cfg.fashion_elements[best_idx],
            float(scores[best_idx]),
        )

    return cluster_labels


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Candidate trend scoring
# ─────────────────────────────────────────────────────────────────────────────

def score_candidate_trends(
    records: list,
    labels: np.ndarray,
    cluster_label_map: dict,
    cfg: Config,
) -> list:
    """
    A cluster qualifies as a candidate trend if its images span
    at least cfg.min_houses_threshold distinct fashion houses.

    Ranked by cross_house_score = image_count x house_count.
    """
    cluster_data = defaultdict(lambda: {"houses": set(), "paths": []})
    for rec, lbl in zip(records, labels):
        if lbl == -1:
            continue
        cluster_data[int(lbl)]["houses"].add(rec.house)
        cluster_data[int(lbl)]["paths"].append(rec.path)

    candidates = []
    for cid, data in cluster_data.items():
        house_count = len(data["houses"])
        if house_count < cfg.min_houses_threshold:
            continue

        label, score = cluster_label_map.get(cid, ("unknown", 0.0))
        image_count = len(data["paths"])
        candidates.append(
            CandidateTrend(
                cluster_id=cid,
                top_label=label,
                label_score=round(score, 4),
                image_count=image_count,
                house_count=house_count,
                houses=sorted(data["houses"]),
                cross_house_score=round(image_count * house_count, 2),
                sample_image_paths=data["paths"][:5],
            )
        )

    candidates.sort(key=lambda c: c.cross_house_score, reverse=True)
    logger.info(
        f"{len(candidates)} candidate trends found "
        f"(min_houses_threshold={cfg.min_houses_threshold})."
    )
    return candidates


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 — Save outputs
# ─────────────────────────────────────────────────────────────────────────────

def save_outputs(
    candidates: list,
    embeddings: np.ndarray,
    labels: np.ndarray,
    cfg: Config,
):
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Main output consumed by demand_side_analysis.py
    trends_path = out / "candidate_trends.json"
    with open(trends_path, "w") as f:
        json.dump([asdict(c) for c in candidates], f, indent=2)
    logger.info(f"Saved candidate trends -> {trends_path}")

    # Raw arrays for visualisation / further analysis
    np.save(out / "embeddings.npy", embeddings)
    np.save(out / "cluster_labels.npy", labels)

    # Human-readable summary
    summary_path = out / "trend_summary.txt"
    with open(summary_path, "w") as f:
        f.write("SUPPLY-SIDE CANDIDATE TRENDS\n")
        f.write("=" * 60 + "\n\n")
        for rank, c in enumerate(candidates, 1):
            f.write(f"#{rank}  {c.top_label.upper()}\n")
            f.write(f"     Confidence        : {c.label_score:.3f}\n")
            f.write(f"     Images            : {c.image_count}\n")
            f.write(
                f"     Houses ({c.house_count:2d})      : "
                f"{', '.join(c.houses)}\n"
            )
            f.write(f"     Cross-house score : {c.cross_house_score}\n\n")
    logger.info(f"Saved summary -> {summary_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main entrypoint
# ─────────────────────────────────────────────────────────────────────────────

def run(cfg: Config):
    logger.info(f"Device: {cfg.device}  |  CLIP model: {cfg.clip_model}")

    records    = load_images(cfg)
    embeddings = extract_embeddings(records, cfg)
    reduced    = reduce_dimensions(embeddings, cfg)
    labels     = cluster_embeddings(reduced, cfg)

    for rec, lbl in zip(records, labels):
        rec.cluster_id = int(lbl)

    cluster_label_map = label_clusters(records, labels, cfg)

    for rec in records:
        if rec.cluster_id is not None and rec.cluster_id in cluster_label_map:
            rec.top_label, rec.label_score = cluster_label_map[rec.cluster_id]

    candidates = score_candidate_trends(records, labels, cluster_label_map, cfg)
    save_outputs(candidates, embeddings, labels, cfg)
    return candidates


if __name__ == "__main__":
    cfg = Config()
    candidates = run(cfg)

    print("\n── TOP 5 CANDIDATE TRENDS ──────────────────────────────")
    for i, c in enumerate(candidates[:5], 1):
        print(
            f"  {i}. {c.top_label}"
            f"  |  {c.house_count} houses"
            f"  |  score: {c.cross_house_score}"
        )
