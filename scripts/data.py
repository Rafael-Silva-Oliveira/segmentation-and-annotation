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
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import sopa
import spatialdata as sd
import yaml
from loguru import logger

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
try:
    SCRIPT_DIR = (
        Path(__file__).resolve().parent
    )
except NameError:
    # Running in a notebook
    SCRIPT_DIR = Path(
        r"C:\Users\rafae\Projects\segmentation-and-annotation\scripts"
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

OUTS_DIR = (
    SPACERANGER_OUTS
    / sample_id
    / "outs"
).resolve()

# %%
sdata = sopa.io.visium_hd(
    r"C:\Users\rafae\Projects\segmentation-and-annotation\data\spatial\SpaceRanger\Visium_HD_Human_Colon_Cancer\outs",
    dataset_id="Visium_HD_Human_Colon_Cancer",
    fullres_image_file=str(
        PROJECT_DIR
        / "data"
        / "spatial"
        / "Microscopy"
        / "Visium_HD_Human_Colon_Cancer_tissue_image.btf"
    ),
)


# %%
# TODO: make it smaller, to run segmentation faster
# Crop to a smaller region for faster processing
sdata_sub = sd.bounding_box_query(
    sdata,
    min_coordinate=[43000, 2000],
    max_coordinate=[60000, 19000],
    axes=("x", "y"),
    target_coordinate_system="Visium_HD_Human_Colon_Cancer",
    filter_table=True,
)

# %%
import spatialdata_plot

# Plot the cropped region
sdata_sub.pl.render_images(
    "Visium_HD_Human_Colon_Cancer_full_image"
).pl.show(
    coordinate_systems="Visium_HD_Human_Colon_Cancer",
    figsize=(10, 10),
)


# %%
for (
    key,
    table,
) in sdata_sub.tables.items():
    table.var_names_make_unique()

# %%
# Patch-based cell segmentation
logger.info("Creating image patches...")
sopa.make_image_patches(sdata_sub)

logger.info(
    "Running StarDist segmentation..."
)
sopa.segmentation.stardist(
    sdata_sub, min_area=20
)

logger.info(
    "Running ProSeg refinement..."
)
sopa.segmentation.proseg(
    sdata_sub, prior_shapes_key="auto"
)

logger.info(
    "Aggregating gene expression to cells..."
)
sopa.aggregate(
    sdata_sub,
    aggregate_channels=False,
    expand_radius_ratio=1,
)

# %%
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
logger.info(
    f"AnnData shape before filtering: {adata.shape}"
)
sc.pp.filter_genes(
    adata,
    min_counts=params[
        "filter_genes_counts"
    ],
)
sc.pp.filter_cells(
    adata,
    min_counts=params[
        "filter_cells_counts"
    ],
)
logger.info(
    f"AnnData shape after filtering: {adata.shape}"
)

# QC metrics
vmax_pct = 99
adata.var["mt"] = (
    adata.var_names.str.startswith(
        ("MT-", "mt-")
    )
)
sc.pp.calculate_qc_metrics(
    adata,
    inplace=True,
    percent_top=None,
)
sc.pp.calculate_qc_metrics(
    adata,
    qc_vars=["mt"],
    inplace=True,
    percent_top=None,
)

# Plot n_counts embedding
fig = sc.pl.embedding(
    adata,
    basis="spatial",
    color="n_counts",
    vmin=np.percentile(
        adata.obs["n_counts"], 1
    ),
    vmax=np.percentile(
        adata.obs["n_counts"], vmax_pct
    ),
    size=8,
    show=False,
    return_fig=True,
)
fig.savefig(
    f"{run_dir}/n_counts_embedding_{sample_id}.png",
    bbox_inches="tight",
    dpi=150,
)
plt.close(fig)

# Plot mito embedding
fig = sc.pl.embedding(
    adata,
    basis="spatial",
    color="pct_counts_mt",
    vmin=np.percentile(
        adata.obs["pct_counts_mt"], 1
    ),
    vmax=np.percentile(
        adata.obs["pct_counts_mt"],
        vmax_pct,
    ),
    size=8,
    show=False,
    return_fig=True,
)
fig.savefig(
    f"{run_dir}/pct_counts_mt_embedding_{sample_id}.png",
    bbox_inches="tight",
    dpi=150,
)
plt.close(fig)

# Filter high mito cells
percentile_pct_mito = params[
    "percentile_pct_mito"
]
pct_mito = params["pct_mito"]
if percentile_pct_mito is not None:
    pct_to_remove = adata.obs[
        "pct_counts_mt"
    ].quantile(percentile_pct_mito)
else:
    pct_to_remove = pct_mito

adata = adata[
    adata.obs["pct_counts_mt"]
    < pct_to_remove
].copy()
logger.info(
    f"After mito filtering ({pct_to_remove}%): {adata.shape}"
)

# Plot mito embedding after filtering
fig = sc.pl.embedding(
    adata,
    basis="spatial",
    color="pct_counts_mt",
    size=8,
    show=False,
    return_fig=True,
)
fig.savefig(
    f"{run_dir}/mt_embedding_after_filtering_{sample_id}.png",
    bbox_inches="tight",
    dpi=150,
)
plt.close(fig)

# Compute spatial neighbors
logger.info(
    "Computing spatial neighbors."
)
spatial_radius = params["radius"]
novae.spatial_neighbors(
    [adata],
    radius=spatial_radius,
    slide_key="sample_id",
    coord_type="generic",
)

# Plot connectivities
novae.plot.connectivities([adata])
plt.savefig(
    f"{run_dir}/connectivities.png",
    dpi=600,
    bbox_inches="tight",
)
plt.close()

# Novae preprocessing (normalize, log1p, HVGs)
logger.info("Preprocessing with Novae")
novae.utils.prepare_adatas([adata])
adata.layers["lognorm_counts"] = (
    adata.X.copy()
)

# PCA
logger.info("Running PCA")
sc.pp.pca(
    adata, use_highly_variable=True
)

# Novae clustering
logger.info(
    "Fine-tuning Novae model..."
)
model = novae.Novae.from_pretrained(
    params["novae_model"]
)
model.fine_tune(
    [adata],
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
model.compute_representations([adata])

# Assign domains across a range of resolutions
domain_min, domain_max = params[
    "domain_range"
]
for n_domains in range(
    domain_min, domain_max + 1
):
    col = f"novae_domains_{n_domains}"
    model.assign_domains(
        [adata], level=n_domains
    )
    model.batch_effect_correction(
        [adata], obs_key=col
    )

    adata.obsm[
        f"novae_latent_{n_domains}"
    ] = adata.obsm[
        "novae_latent"
    ].copy()
    adata.obsm[
        f"novae_latent_corrected_{n_domains}"
    ] = adata.obsm[
        "novae_latent_corrected"
    ].copy()

    novae.plot.domains(
        [adata],
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
        [adata], obs_key=col, show=False
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
adata.write_h5ad(
    str(
        run_dir
        / f"{sample_id}_clustered.h5ad"
    )
)
logger.info(
    f"Saved clustered AnnData: {adata.shape}"
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
if "lognorm_counts" in adata.layers:
    adata.X = adata.layers[
        "lognorm_counts"
    ].copy()

# Filter marker genes to those present in the data
marker_genes_filtered = {
    ct: [
        g
        for g in genes
        if g in adata.var_names
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
    adata=adata,
    gene_set=marker_genes_filtered,
    smoothing=True,
    correct_spatial_covariates=True,
    batch_key="sample_id",
)

# Identify top cell type per domain
score_cols = [
    col
    for col in adata.obs.columns
    if col.endswith("_score")
]
logger.info(
    f"Score columns: {score_cols}"
)

# Assign cell type as the highest scoring marker set per cell
score_df = adata.obs[score_cols].copy()
score_df.columns = [
    col.replace("_score", "")
    for col in score_cols
]
adata.obs["cell_type_enrichmap"] = (
    score_df.idxmax(axis=1)
)

# Per-domain majority vote for cleaner labels
domain_ct = adata.obs.groupby(
    clustering_col
)["cell_type_enrichmap"].agg(
    lambda x: x.value_counts().index[0]
)
adata.obs["cell_type_domain"] = (
    adata.obs[clustering_col].map(
        domain_ct
    )
)

logger.info(
    f"Cell type distribution:\n{adata.obs['cell_type_domain'].value_counts()}"
)

# %%
# =============================================================================
# PLOTS
# =============================================================================
logger.info("=== Generating plots ===")

# Spatial embedding colored by cell type
sc.pl.embedding(
    adata,
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
score_by_domain = adata.obs.groupby(
    clustering_col
)[score_cols].mean()
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
    adata,
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
adata.write_h5ad(
    str(
        run_dir
        / f"{sample_id}_annotated.h5ad"
    )
)

logger.info(
    f"Done! All outputs saved to {run_dir}"
)
