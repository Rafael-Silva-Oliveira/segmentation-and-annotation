# %%
# =============================================================================
# Visium HD CRC Tutorial - End-to-end pipeline
# 10x Genomics Human Colorectal Cancer (FFPE)
# Steps: Segmentation (01) -> Clustering (02) -> Annotation (03)
# =============================================================================
from datetime import datetime
from pathlib import Path

import enrichmap as em
import matplotlib as mpl
import matplotlib.pyplot as plt
import novae
import pandas as pd
import scanpy as sc
import seaborn as sns
import sopa
import spatialdata as sd
import yaml
from loguru import logger

from filtering import (
    filter_spatial_anndata,
)
from preprocessing import (
    preprocess_adatas,
)

# Matplotlib defaults
mpl.rcParams["figure.dpi"] = 300
plt.style.use("bmh")
plt.rcParams.update(
    {
        "figure.figsize": (12, 8),
        "axes.facecolor": "white",
        "axes.edgecolor": "black",
    }
)

# %%
# =============================================================================
# PATHS & CONFIG
# =============================================================================
SCRIPT_DIR = (
    Path(__file__).resolve().parent
)
PROJECT_DIR = SCRIPT_DIR.parent
CONFIG_PATH = (
    SCRIPT_DIR
    / "config_crc_tutorial.yaml"
)

with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

SPACERANGER_OUTS = (
    PROJECT_DIR
    / cfg["paths"]["spaceranger_outs"]
)
PROCESSED_DIR = (
    PROJECT_DIR
    / cfg["paths"]["processed"]
)
FIGURES_DIR = (
    PROJECT_DIR
    / cfg["paths"]["figures"]
)
MARKERS_DIR = (
    PROJECT_DIR / "data" / "markers"
)

sample_id = cfg["samples"][0]["id"]
sample_name = cfg["samples"][0]["name"]
params = cfg["params"]
marker_genes = cfg["marker_genes"]

run_name = f"crc_tutorial_{datetime.now().strftime('%d%m%Y_%H%M')}"
run_dir = PROCESSED_DIR / run_name
run_dir.mkdir(
    parents=True, exist_ok=True
)
(run_dir / "figures").mkdir(
    exist_ok=True
)

logger.info(f"Run directory: {run_dir}")
logger.info(
    f"Sample: {sample_id} ({sample_name})"
)

# %%
# =============================================================================
# STEP 1 — SEGMENTATION (mirrors 01_THORA_ST_segmentation.py)
# =============================================================================
logger.info(
    "=== STEP 1: Reading Visium HD data and running segmentation ==="
)

sdata = sopa.io.visium_hd(
    str(
        SPACERANGER_OUTS
        / sample_id
        / "outs"
    ),
    dataset_id="",
)

# Patch-based cell segmentation
logger.info("Creating image patches...")
sopa.make_image_patches(sdata)

logger.info(
    "Running StarDist segmentation..."
)
sopa.segmentation.stardist(
    sdata, min_area=20
)

logger.info(
    "Running ProSeg refinement..."
)
sopa.segmentation.proseg(
    sdata, prior_shapes_key="auto"
)

logger.info(
    "Aggregating gene expression to cells..."
)
sopa.aggregate(
    sdata,
    aggregate_channels=False,
    expand_radius_ratio=1,
)

# Update table metadata
for size, table in sdata.tables.items():
    table.var_names_make_unique()
    table.obs["sample_id"] = sample_id

# Rasterize 2um bins for visualization
sdata["square_002um"].X = sdata[
    "square_002um"
].X.tocsc()
lazy_bins_image = sd.rasterize_bins(
    sdata,
    bins="_square_002um",
    table_name="square_002um",
    row_key="array_row",
    col_key="array_col",
)
sdata["gene_expression_2_um"] = (
    lazy_bins_image
)

# Save segmented data
segmented_path = (
    PROCESSED_DIR
    / f"{sample_id}_segmented.zarr"
)
logger.info(
    f"Saving segmented SpatialData to {segmented_path}"
)
sdata.write(
    str(segmented_path), overwrite=True
)

# %%
# =============================================================================
# STEP 2 — CLUSTERING (mirrors 02_THORA_ST_clustering.py)
# =============================================================================
logger.info(
    "=== STEP 2: Filtering, preprocessing, and Novae clustering ==="
)

# Extract the cell-level AnnData table
adata = sdata["table"].copy()

# Filter genes and cells
adatas = filter_spatial_anndata(
    concatenated_adata=adata,
    sample_list_names=[sample_id],
    filter_cells_counts=params[
        "filter_cells_counts"
    ],
    filter_genes_counts=params[
        "filter_genes_counts"
    ],
    slide_key="sample_id",
)

# Preprocess: QC, mito filtering, spatial neighbors, HVGs
adatas, adata_concat = (
    preprocess_adatas(
        adatas=adatas,
        spatial_radius=params["radius"],
        slide_key="sample_id",
        sample_list_names=[sample_id],
        use_highly_variable_genes=True,
        run_dir=str(run_dir),
        percentile_pct_mito=params[
            "percentile_pct_mito"
        ],
        pct_mito=params["pct_mito"],
        vmax=99,
    )
)

# Novae clustering
logger.info(
    "Fine-tuning Novae model..."
)
model = novae.Novae.from_pretrained(
    params["novae_model"]
)
model.fine_tune(
    adatas,
    max_epochs=params[
        "novae_max_epochs"
    ],
)
model.save_pretrained(
    str(run_dir / "model")
)

logger.info(
    "Computing Novae representations..."
)
model.compute_representations(adatas)

# Assign domains across a range of resolutions
domain_min, domain_max = params[
    "domain_range"
]
for n_domains in range(
    domain_min, domain_max + 1
):
    col = f"novae_domains_{n_domains}"
    model.assign_domains(
        adatas, level=n_domains
    )
    model.batch_effect_correction(
        adatas, obs_key=col
    )

    for data in adatas:
        data.obsm[
            f"novae_latent_{n_domains}"
        ] = data.obsm[
            "novae_latent"
        ].copy()
        data.obsm[
            f"novae_latent_corrected_{n_domains}"
        ] = data.obsm[
            "novae_latent_corrected"
        ].copy()

    novae.plot.domains(
        adatas,
        slide_name_key="sample_id",
        cell_size=8,
        show=False,
        obs_key=col,
    )
    plt.savefig(
        str(
            run_dir
            / f"domains_{n_domains}.png"
        ),
        dpi=600,
        bbox_inches="tight",
    )
    plt.close()

    novae.plot.domains_proportions(
        adatas, obs_key=col, show=False
    )
    plt.savefig(
        str(
            run_dir
            / f"domains_proportions_{n_domains}.png"
        ),
        dpi=600,
        bbox_inches="tight",
    )
    plt.close()

# Save clustered AnnData
adata_clustered = adatas[0]
adata_clustered.write_h5ad(
    str(
        run_dir
        / f"{sample_id}_clustered.h5ad"
    )
)
logger.info(
    f"Saved clustered AnnData: {adata_clustered.shape}"
)

# %%
# =============================================================================
# STEP 3 — ANNOTATION (mirrors 03_THORA_ST_annotation.py)
# =============================================================================
logger.info(
    "=== STEP 3: Cell type annotation with Enrichmap ==="
)

# Load marker genes from CSV files in data/markers/ and merge with config markers
logger.info(
    f"Loading marker CSVs from {MARKERS_DIR}"
)
for csv_file in MARKERS_DIR.glob(
    "*.csv"
):
    df = pd.read_csv(csv_file)
    cell_type = csv_file.stem.replace(
        "_", " "
    )
    genes_from_csv = df["Name"].tolist()
    # Merge CSV markers into config markers (config takes precedence, CSV adds extras)
    config_key = csv_file.stem
    if config_key in marker_genes:
        existing = set(
            marker_genes[config_key]
        )
        for g in genes_from_csv:
            if g not in existing:
                marker_genes[
                    config_key
                ].append(g)
    else:
        marker_genes[config_key] = (
            genes_from_csv
        )
    logger.info(
        f"  {config_key}: {marker_genes[config_key]}"
    )

# Use the chosen clustering resolution for annotation
clustering_col = params[
    "clustering_col"
]
logger.info(
    f"Annotating using clustering column: {clustering_col}"
)

# Ensure we use raw counts for enrichmap (lognorm layer was saved during preprocessing)
if (
    "lognorm_counts"
    in adata_clustered.layers
):
    adata_clustered.X = (
        adata_clustered.layers[
            "lognorm_counts"
        ].copy()
    )

# Filter marker genes to those present in the data
marker_genes_filtered = {
    ct: [
        g
        for g in genes
        if g
        in adata_clustered.var_names
    ]
    for ct, genes in marker_genes.items()
}
marker_genes_filtered = {
    ct: genes
    for ct, genes in marker_genes_filtered.items()
    if len(genes) > 0
}
logger.info(
    f"Marker genes after filtering to var_names: { {k: len(v) for k, v in marker_genes_filtered.items()} }"
)

# Run Enrichmap scoring
logger.info(
    "Running Enrichmap scoring..."
)
em.tl.score(
    adata=adata_clustered,
    gene_set=marker_genes_filtered,
    smoothing=True,
    correct_spatial_covariates=True,
    batch_key="sample_id",
)

# Identify top cell type per domain
score_cols = [
    col
    for col in adata_clustered.obs.columns
    if col.endswith("_score")
]
logger.info(
    f"Score columns: {score_cols}"
)

# Assign cell type as the highest scoring marker set per cell
score_df = adata_clustered.obs[
    score_cols
].copy()
score_df.columns = [
    col.replace("_score", "")
    for col in score_cols
]
adata_clustered.obs[
    "cell_type_enrichmap"
] = score_df.idxmax(axis=1)

# Per-domain majority vote for cleaner labels
domain_ct = adata_clustered.obs.groupby(
    clustering_col
)["cell_type_enrichmap"].agg(
    lambda x: x.value_counts().index[0]
)
adata_clustered.obs[
    "cell_type_domain"
] = adata_clustered.obs[
    clustering_col
].map(domain_ct)

logger.info(
    f"Cell type distribution:\n{adata_clustered.obs['cell_type_domain'].value_counts()}"
)

# %%
# =============================================================================
# PLOTS
# =============================================================================
logger.info("=== Generating plots ===")

# Spatial embedding colored by cell type
sc.pl.embedding(
    adata_clustered,
    basis="spatial",
    color="cell_type_domain",
    size=8,
    show=False,
    title=f"{sample_name} - Cell Types",
)
plt.savefig(
    str(
        run_dir
        / "cell_types_spatial.png"
    ),
    dpi=600,
    bbox_inches="tight",
)
plt.close()

# Enrichmap score heatmap per domain
score_by_domain = (
    adata_clustered.obs.groupby(
        clustering_col
    )[score_cols].mean()
)
score_by_domain.columns = [
    col.replace("_score", "")
    for col in score_by_domain.columns
]

fig, ax = plt.subplots(
    figsize=(
        max(
            10,
            len(score_by_domain.columns)
            * 0.6,
        ),
        len(score_by_domain) * 0.4 + 2,
    )
)
sns.heatmap(
    score_by_domain,
    cmap="RdBu_r",
    center=0,
    ax=ax,
    linewidths=0.5,
)
ax.set_title(
    "Enrichmap Scores per Novae Domain"
)
ax.set_ylabel("Domain")
plt.tight_layout()
plt.savefig(
    str(
        run_dir
        / "enrichmap_heatmap.png"
    ),
    dpi=300,
    bbox_inches="tight",
)
plt.close()

# Dotplot of marker genes by cell type domain
sc.pl.dotplot(
    adata_clustered,
    var_names=marker_genes_filtered,
    groupby="cell_type_domain",
    show=False,
    standard_scale="var",
)
plt.savefig(
    str(
        run_dir / "dotplot_markers.png"
    ),
    dpi=300,
    bbox_inches="tight",
)
plt.close()

# %%
# =============================================================================
# SAVE FINAL RESULTS
# =============================================================================
logger.info(
    "Saving final annotated AnnData..."
)
adata_clustered.write_h5ad(
    str(
        run_dir
        / f"{sample_id}_annotated.h5ad"
    )
)

logger.info(
    f"Done! All outputs saved to {run_dir}"
)
