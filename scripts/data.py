# %%
# =============================================================================
# Visium HD CRC Tutorial - End-to-end pipeline
# 10x Genomics Human Colorectal Cancer (FFPE)
# Steps: Segmentation (01) -> Clustering (02) -> Annotation (03)
# =============================================================================
import warnings
from datetime import datetime
from pathlib import Path

import decoupler as dc
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
import spatialdata_plot
import yaml
import zarr
from anndata import AnnData
from loguru import logger
import spatialdata_plot

warnings.filterwarnings(
    "ignore",
    message="Use `squidpy.pl.spatial_scatter`",
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
PROJECT_DIR = Path(
    r"C:\Users\rafae\Projects\segmentation-and-annotation"
).resolve()
CONFIG_PATH = (
    PROJECT_DIR
    / "scripts"
    / "config_crc_tutorial.yaml"
)

with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

sample_id = cfg["samples"][0]["id"]
sample_name = cfg["samples"][0]["name"]
params = cfg["params"]
marker_genes = cfg["marker_genes"]

PROCESSED_DIR = (
    PROJECT_DIR
    / "data"
    / "spatial"
    / "processed"
).resolve()
FIGURES_DIR = (
    PROJECT_DIR / "figures"
).resolve()

run_name = f"crc_tutorial_{datetime.now().strftime('%d%m%Y_%H%M')}"
run_dir = (
    PROCESSED_DIR / run_name
).resolve()
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
# STEP 1 — SEGMENTATION
# =============================================================================
logger.info(
    "=== STEP 1: Reading Visium HD data and running segmentation ==="
)

sdata = sopa.io.visium_hd(
    str(
        PROJECT_DIR
        / "data"
        / "spatial"
        / "SpaceRanger"
        / sample_id
        / "outs"
    ),
    dataset_id=sample_id,
    fullres_image_file=str(
        PROJECT_DIR
        / "data"
        / "spatial"
        / "Microscopy"
        / "Visium_HD_Human_Colon_Cancer_tissue_image.btf"
    ),
)

# %%
# Crop to a region with tumor, stroma, and normal mucosa
sdata_sub = sd.bounding_box_query(
    sdata,
    min_coordinate=[51000, 9000],
    max_coordinate=[56000, 14000],
    axes=("x", "y"),
    target_coordinate_system=sample_id,
    filter_table=True,
)

# sdata_sub.write("test.zarr")  # save it

# %%
# Plot the cropped region
sdata_sub.pl.render_images(
    f"{sample_id}_full_image"
).pl.show(
    coordinate_systems=sample_id,
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
    sdata_sub,
    prior_shapes_key="stardist_boundaries",
)

# %%
logger.info(
    "Aggregating gene expression to cells..."
)
sopa.aggregate(
    sdata_sub,
    aggregate_channels=False,
    expand_radius_ratio=1,
)

# %%

# Plot segmentation overlay
sdata_sub.pl.render_images(
    f"{sample_id}_full_image"
).pl.render_shapes(
    "stardist_boundaries",
    outline=True,
    fill_alpha=0,
    outline_alpha=0.6,
    outline_color="yellow",
).pl.show(
    coordinate_systems=sample_id,
    figsize=(12, 12),
    title="StarDist Cell Segmentation",
)

# %%
# Rasterize 2um bins for visualization
sdata_sub["square_002um"].X = sdata_sub[
    "square_002um"
].X.tocsc()
lazy_bins_image = sd.rasterize_bins(
    sdata_sub,
    bins="_square_002um",
    table_name="square_002um",
    row_key="array_row",
    col_key="array_col",
)
sdata_sub["gene_expression_2_um"] = (
    lazy_bins_image
)

# %%

# Save segmented data
segmented_h5ad = (
    PROJECT_DIR
    / "data"
    / "spatial"
    / "segmented.h5ad"
)
sdata_sub.tables["table"].write_h5ad(
    str(segmented_h5ad)
)
logger.info(
    f"Saved segmented table to {segmented_h5ad}"
)

# %%
# =============================================================================
# STEP 2 — CLUSTERING
# =============================================================================
logger.info(
    "=== STEP 2: Filtering, preprocessing, and Novae clustering ==="
)

adata = sc.read_h5ad(
    str(segmented_h5ad)
)

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

# %%
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

# Plot n_counts
fig = sc.pl.spatial(
    adata,
    color="n_counts",
    vmin=np.percentile(
        adata.obs["n_counts"], 1
    ),
    vmax=np.percentile(
        adata.obs["n_counts"], vmax_pct
    ),
    spot_size=10,
    show=False,
    return_fig=True,
)
fig.savefig(
    f"{run_dir}/n_counts_{sample_id}.png",
    bbox_inches="tight",
    dpi=150,
)
plt.close(fig)

# Plot mito percentage
fig = sc.pl.spatial(
    adata,
    color="pct_counts_mt",
    vmin=np.percentile(
        adata.obs["pct_counts_mt"], 1
    ),
    vmax=np.percentile(
        adata.obs["pct_counts_mt"],
        vmax_pct,
    ),
    spot_size=10,
    show=False,
    return_fig=True,
)
fig.savefig(
    f"{run_dir}/pct_counts_mt_{sample_id}.png",
    bbox_inches="tight",
    dpi=150,
)
plt.close(fig)

# Filter high mito cells
pct_to_remove = params["pct_mito"]
if (
    params["percentile_pct_mito"]
    is not None
):
    pct_to_remove = adata.obs[
        "pct_counts_mt"
    ].quantile(
        params["percentile_pct_mito"]
    )

adata = adata[
    adata.obs["pct_counts_mt"]
    < pct_to_remove
].copy()
logger.info(
    f"After mito filtering ({pct_to_remove}%): {adata.shape}"
)

# Plot mito after filtering
fig = sc.pl.spatial(
    adata,
    color="pct_counts_mt",
    spot_size=10,
    show=False,
    return_fig=True,
)
fig.savefig(
    f"{run_dir}/mt_after_filtering_{sample_id}.png",
    bbox_inches="tight",
    dpi=150,
)
plt.close(fig)

# %%
# Compute spatial neighbors
logger.info(
    "Computing spatial neighbors."
)
novae.spatial_neighbors(
    [adata],
    radius=params["radius"],
    coord_type="generic",
)

novae.plot.connectivities([adata])
plt.savefig(
    f"{run_dir}/connectivities.png",
    dpi=600,
    bbox_inches="tight",
)
plt.close()

# %%
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
# NOTE: train our own model. See more on Novae page (put link)
model.fine_tune(
    [adata],
    max_epochs=params[
        "novae_max_epochs"
    ],
)
model.save_pretrained(
    str(run_dir / "model")
)

# %%
logger.info(
    "Computing Novae representations..."
)
# Inference based on zero_shot
model.compute_representations(
    [adata], zero_shot=True
)

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

    adata.obsm[
        f"novae_latent_{n_domains}"
    ] = adata.obsm[
        "novae_latent"
    ].copy()

    novae.plot.domains(
        [adata],
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

# %%
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
# STEP 3 — ANNOTATION
# =============================================================================

adata = sc.read_h5ad(
    r"C:\Users\rafae\Projects\segmentation-and-annotation\data\spatial\processed\crc_tutorial_17032026_1134\Visium_HD_Human_Colon_Cancer_clustered.h5ad"
)
logger.info(
    "=== STEP 3: Cell type annotation with Enrichmap ==="
)

clustering_col = "novae_domains_8"
logger.info(
    f"Annotating using clustering column: {clustering_col}"
)
logger.info(
    f"Marker genes: {marker_genes}"
)

# Ensure we use lognorm counts for enrichmap
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
    f"Marker genes after filtering: { {k: len(v) for k, v in marker_genes_filtered.items()} }"
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
)

# %% We can do this or use em.pl.spatial_enrichmap()
import matplotlib.pyplot as plt
import numpy as np

coords = adata.obsm["spatial"]
scores = adata.obs["Neutrophils_score"]

fig, ax = plt.subplots(figsize=(10, 10))
scatter = ax.scatter(
    coords[:, 0],
    coords[:, 1],
    c=scores,
    cmap="seismic",
    s=3,
    vmin=-scores.abs().max(),
    vmax=scores.abs().max(),
)
ax.set_aspect("equal")
ax.invert_yaxis()
ax.axis("off")
ax.set_title("Tumor")
plt.colorbar(scatter, ax=ax, shrink=0.5)
plt.show()


# %%
# Rank cell type scores per domain using decoupler (1-vs-rest Wilcoxon)
score_cols = [
    col
    for col in adata.obs.columns
    if col.endswith("_score")
]
logger.info(
    f"Score columns: {score_cols}"
)

score_matrix = adata.obs[
    score_cols
].values
score_adata = AnnData(
    X=score_matrix,
    obs=adata.obs[
        [clustering_col]
    ].copy(),
)
score_adata.var_names = [
    col.replace("_score", "")
    for col in score_cols
]

# NOTE: Mention that we can either do 1 vs rest test using decoupler or scanpy or other testing method, or simply do an idxmax on the enrichmap scores and assign that given segmented cell to the given cell type. In this tutorial, I am showing how to do the rankby_group method using dc. The reason is that we can then use an elbow gap method later on to find spots that may be capturing more heterogeneity (more than 1 cell type) and attribute to 1 or more cell types for a given segmented cell.
ranking_df = dc.tl.rankby_group(
    score_adata,
    groupby=clustering_col,
    reference="rest",
    method="wilcoxon",
)

# %%
ranking_df = ranking_df.sort_values(
    by=["group", "meanchange"],
    ascending=[True, False],
)
logger.info(
    f"Ranking shape: {ranking_df.shape}"
)
ranking_df.to_csv(
    str(
        run_dir
        / f"enrichment_ranking_{clustering_col}.csv"
    ),
    index=False,
)

# %%
# Gap-based annotation: for each domain, find the biggest drop
# in meanchange among significant cell types
enrichmap_gap_annotations = {}
elbow_data = {}

for domain in ranking_df[
    "group"
].unique():
    domain_data = ranking_df.loc[
        ranking_df["group"] == domain
    ].copy()

    # Filter to significant, positively enriched
    domain_data = domain_data.loc[
        (domain_data["meanchange"] > 0)
        & (domain_data["padj"] < 0.05)
    ]

    if len(domain_data) == 0:
        enrichmap_gap_annotations[
            domain
        ] = "Unknown"
        continue

    if len(domain_data) == 1:
        enrichmap_gap_annotations[
            domain
        ] = domain_data["name"].iloc[0]
        elbow_data[domain] = domain_data
        continue

    elbow_data[domain] = domain_data

    sorted_mc = domain_data[
        "meanchange"
    ].values
    gaps = (
        sorted_mc[:-1] - sorted_mc[1:]
    )
    biggest_gap_idx = int(
        np.argmax(gaps)
    )

    selected = (
        domain_data["name"]
        .iloc[: biggest_gap_idx + 1]
        .tolist()
    )
    enrichmap_gap_annotations[
        domain
    ] = "/".join(selected)

logger.info(
    f"Domain -> cell type mapping:\n{enrichmap_gap_annotations}"
)

# Assign domain-level cell type
adata.obs["cell_type_domain"] = (
    adata.obs[clustering_col].map(
        enrichmap_gap_annotations
    )
)

# Per-cell high granularity: argmax over scores
score_df = adata.obs[score_cols].copy()
score_df.columns = [
    col.replace("_score", "")
    for col in score_cols
]
adata.obs["cell_type_enrichmap"] = (
    score_df.idxmax(axis=1)
)

logger.info(
    f"Cell type distribution:\n{adata.obs['cell_type_domain'].value_counts()}"
)

# %%
# Elbow plot: meanchange vs ranked cell types per domain
plottable_domains = [
    d
    for d in elbow_data
    if len(elbow_data[d]) > 0
]
n_domains_plot = len(plottable_domains)
n_cols_plot = min(4, n_domains_plot)
n_rows_plot = int(
    np.ceil(
        n_domains_plot / n_cols_plot
    )
)

fig_elbow, axes_elbow = plt.subplots(
    n_rows_plot,
    n_cols_plot,
    figsize=(
        5 * n_cols_plot,
        4 * n_rows_plot,
    ),
)
axes_flat = np.array(
    axes_elbow
).flatten()

for ax_idx, domain in enumerate(
    plottable_domains
):
    ax = axes_flat[ax_idx]
    domain_data = elbow_data[domain]
    mc_vals = domain_data[
        "meanchange"
    ].values
    ct_names = domain_data[
        "name"
    ].values
    x_pos = np.arange(len(mc_vals))

    selected_names = set(
        enrichmap_gap_annotations[
            domain
        ].split("/")
    )

    ax.plot(
        x_pos,
        mc_vals,
        color="gray",
        lw=1.5,
        zorder=2,
    )

    for i, (xp, mc, name) in enumerate(
        zip(x_pos, mc_vals, ct_names)
    ):
        if name in selected_names:
            color = "steelblue"
            ec = "black"
        else:
            color = "silver"
            ec = "gray"
        ax.scatter(
            xp,
            mc,
            color=color,
            s=60,
            zorder=4,
            edgecolors=ec,
            linewidths=0.5,
        )

    if len(mc_vals) > 1:
        gaps = (
            mc_vals[:-1] - mc_vals[1:]
        )
        gap_idx = int(np.argmax(gaps))
        gap_x = gap_idx + 0.5
        gap_val = gaps[gap_idx]
        ax.axvline(
            gap_x,
            color="red",
            ls="--",
            lw=1,
            alpha=0.8,
            label=f"gap = {gap_val:.3f}",
        )
        ax.axvspan(
            -0.5,
            gap_x,
            alpha=0.08,
            color="steelblue",
        )
        y_top = mc_vals[gap_idx]
        y_bot = mc_vals[gap_idx + 1]
        bracket_x = gap_x + 0.3
        ax.annotate(
            "",
            xy=(bracket_x, y_bot),
            xytext=(bracket_x, y_top),
            arrowprops=dict(
                arrowstyle="<->",
                color="red",
                lw=1.2,
            ),
        )
        ax.text(
            bracket_x + 0.15,
            (y_top + y_bot) / 2,
            f"{gap_val:.3f}",
            fontsize=6,
            color="red",
            va="center",
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        ct_names,
        rotation=60,
        ha="right",
        fontsize=5,
    )
    ax.set_ylabel(
        "meanchange", fontsize=7
    )
    ax.set_title(
        f"Domain {domain}", fontsize=8
    )
    ax.legend(
        fontsize=5, loc="upper right"
    )
    ax.tick_params(labelsize=6)

for ax_idx in range(
    n_domains_plot, len(axes_flat)
):
    axes_flat[ax_idx].axis("off")

fig_elbow.suptitle(
    f"Gap-based elbow — {clustering_col}",
    fontsize=11,
)
plt.tight_layout()
plt.savefig(
    str(
        run_dir
        / f"elbow_plot_{clustering_col}.png"
    ),
    dpi=400,
    bbox_inches="tight",
)
plt.close()
logger.info("Saved elbow plot")

# %%
# =============================================================================
# PLOTS
# =============================================================================
logger.info("=== Generating plots ===")

cell_type_palette = {
    "Tumor": "#E64B35",
    "Fibroblasts": "#4DBBD5",
    "Macrophages": "#3C5488",
    "Neutrophils": "#F39B7F",
    "Goblet_cells": "#00599F",
    "Macrophages/Neutrophils": "#FFD700",
    "Unknown": "#B09C85",
}

sc.pl.spatial(
    adata,
    color="cell_type_domain",
    spot_size=9,
    show=False,
    title=f"{sample_name} - Cell Types",
    palette=[
        cell_type_palette.get(
            ct, "#999999"
        )
        for ct in adata.obs[
            "cell_type_domain"
        ].cat.categories
    ],
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

# %%

adata = sc.read_h5ad(
    r"C:\Users\rafae\Projects\segmentation-and-annotation\data\processed\crc_tutorial_17032026_1255\Visium_HD_Human_Colon_Cancer_annotated.h5ad"
)
sc.pl.spatial(
    adata,
    color="novae_domains_8",
    spot_size=9,
    show=False,
    title="Novae Domains (8)",
)
plt.savefig(
    str(
        run_dir
        / "novaoe_domains_spatial.png"
    ),
    dpi=600,
    bbox_inches="tight",
)
plt.close()

# %%
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


# %%
adata = sc.read_h5ad(
    r"C:\Users\rafae\Projects\segmentation-and-annotation\data\spatial\processed\crc_tutorial_17032026_1255\Visium_HD_Human_Colon_Cancer_annotated.h5ad"
)

# %%


# Helper: map point coordinates from adata.obsm['spatial'] into an image
# coordinate system, create a PointsModel, attach it to `sdata_sub`, and
# optionally plot it. Returns diagnostics and the mapped DataFrame.
def map_and_plot_points(
    adata,
    sdata_sub,
    sample_id,
    color_col="novae_domains_8",
    points_key="segmented_points",
    size=6,
    alpha=0.7,
    transformations=None,
    show_plot=True,
):
    """Map local adata spatial coords into the sample image coordinate
    system, create a PointsModel and attach to sdata_sub, and plot.

    Returns a dict with diagnostics and the mapped DataFrame.
    """
    import numpy as _np
    import pandas as _pd
    from spatialdata.models import (
        PointsModel as _PointsModel,
        TableModel as _TableModel,
    )
    from spatialdata.transformations import (
        Identity as _Identity,
    )

    # 1. Prepare AnnData table metadata
    adata.obs["region"] = (
        "segmented_points"
    )
    adata.obs["region"] = adata.obs[
        "region"
    ].astype("category")
    adata.obs["instance_id"] = (
        adata.obs_names
    )

    _ = _TableModel.parse(
        adata,
        region="segmented_points",
        region_key="region",
        instance_key="instance_id",
        overwrite_metadata=True,
    )
    sdata_sub.tables[
        "segmented_table"
    ] = adata

    # 2. Build DataFrame from local spatial coordinates
    coords = _np.asarray(
        adata.obsm["spatial"]
    )
    df_local = _pd.DataFrame(
        coords,
        columns=["x", "y"],
        index=adata.obs_names,
    )

    # 3. Map local coordinates into the sample image coordinate system
    bbox = sd.get_extent(
        sdata_sub,
        elements=[
            f"{sample_id}_full_image"
        ],
        coordinate_system=sample_id,
    )
    bx0 = float(bbox["x"][0])
    bx1 = float(bbox["x"][1])
    by0 = float(bbox["y"][0])
    by1 = float(bbox["y"][1])

    local_x_min = float(
        df_local["x"].min()
    )
    local_x_max = float(
        df_local["x"].max()
    )
    local_y_min = float(
        df_local["y"].min()
    )
    local_y_max = float(
        df_local["y"].max()
    )

    local_x_span = (
        local_x_max - local_x_min
    )
    local_y_span = (
        local_y_max - local_y_min
    )
    bbox_x_span = bx1 - bx0
    bbox_y_span = by1 - by0

    scale_x = (
        bbox_x_span / local_x_span
        if local_x_span > 0
        else 1.0
    )
    scale_y = (
        bbox_y_span / local_y_span
        if local_y_span > 0
        else 1.0
    )

    df_mapped = _pd.DataFrame(
        index=df_local.index
    )
    df_mapped["x"] = (
        bx0
        + (
            df_local["x"].astype(float)
            - local_x_min
        )
        * scale_x
    )
    df_mapped["y"] = (
        by0
        + (
            df_local["y"].astype(float)
            - local_y_min
        )
        * scale_y
    )

    def _frac_inside(dframe):
        in_bbox = (
            (dframe["x"] >= bx0)
            & (dframe["x"] <= bx1)
            & (dframe["y"] >= by0)
            & (dframe["y"] <= by1)
        )
        return in_bbox.sum() / len(
            dframe
        )

    frac_mapped = _frac_inside(
        df_mapped
    )
    # y-flip variant
    df_mapped_flip = df_mapped.copy()
    df_mapped_flip["y"] = by1 - (
        df_mapped["y"] - by0
    )
    frac_flip = _frac_inside(
        df_mapped_flip
    )

    used_flip = False
    if frac_flip > frac_mapped:
        df = df_mapped_flip
        used_flip = True
    else:
        df = df_mapped

    # 4. Diagnostic: compare to segmentation centroids if available
    centroid_stats = None
    try:
        shape_keys = list(
            sdata_sub.shapes.keys()
        )
        candidate_shape_keys = [
            k
            for k in shape_keys
            if "proseg_boundaries" in k
            or "stardist_boundaries"
            in k
        ]
        if (
            len(candidate_shape_keys)
            > 0
        ):
            shape_key = (
                candidate_shape_keys[0]
            )
            shapes_gdf = (
                sdata_sub.shapes[
                    shape_key
                ]
            )
            centroids = shapes_gdf.geometry.centroid
            centroids_xy = _np.vstack(
                [
                    centroids.x.values,
                    centroids.y.values,
                ]
            ).T
            try:
                from scipy.spatial import (
                    cKDTree as KDTree,
                )

                tree = KDTree(
                    centroids_xy
                )
                dists, _ = tree.query(
                    df[
                        ["x", "y"]
                    ].values,
                    k=1,
                )
            except Exception:
                pts = df[
                    ["x", "y"]
                ].values
                dists = _np.empty(
                    len(pts),
                    dtype=float,
                )
                for i in range(
                    len(pts)
                ):
                    dx = (
                        centroids_xy[
                            :, 0
                        ]
                        - pts[i, 0]
                    )
                    dy = (
                        centroids_xy[
                            :, 1
                        ]
                        - pts[i, 1]
                    )
                    dists[i] = _np.sqrt(
                        (
                            dx * dx
                            + dy * dy
                        ).min()
                    )
            centroid_stats = {
                "median": float(
                    _np.median(dists)
                ),
                "mean": float(
                    _np.mean(dists)
                ),
                "max": float(
                    _np.max(dists)
                ),
            }
    except Exception:
        centroid_stats = None

    # 5. Attach color column and sanitize
    if (
        color_col
        not in adata.obs.columns
    ):
        raise KeyError(
            f"color_col '{color_col}' not found in adata.obs"
        )
    s = adata.obs[color_col].reindex(
        df.index
    )
    s = (
        s.astype(object)
        .fillna("Unknown")
        .astype(str)
    )
    df[color_col] = s.values

    # 6. Create PointsModel and attach
    _transformations = (
        transformations
        if transformations is not None
        else {sample_id: _Identity()}
    )
    points = _PointsModel.parse(
        df,
        coordinates={
            "x": "x",
            "y": "y",
        },
        transformations=_transformations,
    )
    sdata_sub.points[points_key] = (
        points
    )

    # 7. Plot
    if show_plot:
        (
            sdata_sub.pl.render_images(
                f"{sample_id}_full_image"
            )
            .pl.render_points(
                points_key,
                color=color_col,
                size=size,
                alpha=alpha,
                method="matplotlib",
            )
            .pl.show(
                coordinate_systems=sample_id,
                figsize=(12, 12),
                title=f"{color_col} on H&E",
            )
        )

    return {
        "scale_x": scale_x,
        "scale_y": scale_y,
        "frac_in_bbox": float(
            frac_mapped
        ),
        "frac_in_bbox_flip": float(
            frac_flip
        ),
        "used_flip": used_flip,
        "centroid_stats": centroid_stats,
        "df": df,
    }


# %%
# Call the helper to map points and plot (default color column is novae_domains_8)
map_and_plot_points(
    adata,
    sdata_sub,
    sample_id,
    color_col="novae_domains_8",
    points_key="segmented_points",
    size=8,
    alpha=0.7,
)
# %%
map_and_plot_points(
    adata,
    sdata_sub,
    sample_id,
    color_col="cell_type_domain",
    points_key="segmented_points",
    size=8,
    alpha=0.7,
)

# %%
# =============================================================================
# ADDITIONAL STEP: Annotation using reference scRNA dataset with RCTD-py and FlashDeconv
# Reference: GSE200997 - Human CRC scRNA-seq (Lee et al.)
# =============================================================================
import urllib.request, os

ref_url = "https://datasets.cellxgene.cziscience.com/0ca03016-58ec-4d4e-86d9-22dda860bc8c.h5ad"
ref_path = "pelka_crc_all_cells.h5ad"

if not os.path.exists(ref_path):
    print(
        "Downloading Pelka et al. CRC reference (~370k cells)..."
    )
    urllib.request.urlretrieve(
        ref_url, ref_path
    )

adata_ref = sc.read_h5ad(ref_path)
ct_col = "ClusterMidway"


print(
    adata_ref.obs[ct_col].value_counts()
)


# %%
adata = sc.read_h5ad(
    r"C:\Users\rafae\Projects\segmentation-and-annotation\data\processed\crc_tutorial_17032026_1255\Visium_HD_Human_Colon_Cancer_annotated.h5ad"
)

# Subset genes from our segmented cell dataset adata object and our ref, adata_ref
# Load the 008 um data
adata_st_subset = sdata_sub.tables[
    "square_008um"
].copy()

# Re-index reference by gene symbols (CellxGene uses Ensembl IDs as var_names)
adata_ref.var_names = adata_ref.var[
    "feature_name"
].astype(str)
adata_ref.var_names_make_unique()

# Intersect on shared gene symbols across ALL three objects
common_genes = (
    adata.var_names.intersection(
        adata_st_subset.var_names
    ).intersection(adata_ref.var_names)
)
print(
    f"Common genes: {len(common_genes)} "
    f"(adata: {adata.n_vars}, "
    f"spatial: {adata_st_subset.n_vars}, "
    f"ref: {adata_ref.n_vars})"
)

adata = adata[:, common_genes].copy()
adata_st_subset = adata_st_subset[
    :, common_genes
].copy()
adata_ref = adata_ref[
    :, common_genes
].copy()

# %%

# Filter by UMI (same range as rctd-py: 100–20M)
umi = adata_st_subset.X.sum(axis=1).A1

umi_mask = (umi >= 100) & (
    umi <= 20_000_000
)
adata_st_filtered = adata_st_subset[
    umi_mask
].copy()
print(
    f"UMI filter: kept {umi_mask.sum()}/{len(umi_mask)} pixels"
)
# %%
import flashdeconv as fd

# Deconvolve -https://github.com/cafferychen777/flashdeconv
fd.tl.deconvolve(
    adata_st_filtered,
    adata_ref,
    cell_type_key=ct_col,
)

sc.pl.spatial(
    adata_st_filtered,
    color="flashdeconv_dominant",
    spot_size=33,
    show=False,
)
plt.savefig(
    "flashdeconv_dominant.png",
    dpi=2000,
    bbox_inches="tight",
)
plt.close()
# %%

# Downsample to 60k cells, preserving cell type proportions
n_target = 60_000
if adata_ref.n_obs > n_target:
    ct_counts = adata_ref.obs[
        ct_col
    ].value_counts()
    ct_fracs = (
        ct_counts / ct_counts.sum()
    )
    ct_n = (
        (ct_fracs * n_target)
        .round()
        .astype(int)
    )
    # Ensure we don't exceed available cells per type
    ct_n = ct_n.clip(upper=ct_counts)
    idx = []
    for ct, n in ct_n.items():
        ct_idx = adata_ref.obs.index[
            adata_ref.obs[ct_col] == ct
        ]
        idx.extend(
            ct_idx.to_series()
            .sample(
                n=n, random_state=42
            )
            .tolist()
        )
    adata_ref_dw = adata_ref[idx].copy()
    print(
        f"Downsampled ref: {adata_ref_dw.n_obs} cells"
    )

# %%
# https://github.com/p-gueguen/rctd-py
# Disable torch.compile/inductor — requires MSVC (cl) which is not available on this system
import torch._dynamo

torch._dynamo.config.suppress_errors = (
    True
)
torch._dynamo.disable()

from rctd import Reference, run_rctd

# Downsample the reference file to avoid memory errors:
reference = Reference(
    adata_ref_dw,
    cell_type_col=ct_col,
)

# Run RCTD — handles normalization, sigma estimation, and deconvolution
result = run_rctd(
    adata_st_filtered,
    reference,
    mode="full",  # https://p-gueguen.github.io/rctd-py/tutorial.html # doublet or full
)

# %%

# Store per-cell-type weights directly (results align 1:1 with adata_st_filtered)
for i, ct in enumerate(
    result.cell_type_names
):
    adata_st_filtered.obs[
        f"rctd_{ct}"
    ] = result.weights[:, i]

# Dominant cell type per spot
adata_st_filtered.obs[
    "rctd_dominant"
] = pd.Categorical(
    [
        result.cell_type_names[i]
        for i in result.weights.argmax(
            axis=1
        )
    ]
)

sc.pl.spatial(
    adata_st_filtered,
    color="rctd_dominant",
    spot_size=33,
    show=False,
)
plt.savefig(
    "rctd_dominant.png",
    dpi=2000,
    bbox_inches="tight",
)
plt.close()

# %%
# Celltypist
import celltypist

adata.X = adata.layers["counts"].copy()
sc.pp.normalize_total(
    adata,
    target_sum=1e4,
)
sc.pp.log1p(adata)

adata.layers[
    "lognorm_counts_celltypist"
] = adata.X.copy()

# Now do the same log-normalization for the adata_ref
sc.pp.normalize_total(
    adata_ref_dw,
    target_sum=1e4,
)
sc.pp.log1p(adata_ref_dw)

adata_ref_dw.layers[
    "lognorm_counts_celltypist"
] = adata_ref_dw.X.copy()


model = celltypist.train(
    adata_ref_dw,
    labels=ct_col,
    n_jobs=1,
    feature_selection=True,
)

model_path = "celltypist_model.pkl"
model.write(model_path)


# %%

# Load pre-built CellTypist model for Human Colorectal Cancer
celltypist.models.download_models(
    model="Human_Colorectal_Cancer.pkl",
    force_update=False,
)
model = celltypist.models.Model.load(
    model="Human_Colorectal_Cancer.pkl"
)


# %%

col = adata.obs["novae_domains_8"]
adata.obs["novae_domains_8"] = (
    col.cat.add_categories("Unassigned")
    .fillna("Unassigned")
    .astype(str)
)

predictions = celltypist.annotate(
    adata,
    model=model,
    majority_voting=True,
    over_clustering="novae_domains_8",
)
preds_adata = predictions.to_adata()

adata.obs[f"cell_type_celltypist"] = (
    preds_adata.obs[
        "majority_voting"
    ].values
)
adata.obs[
    f"cell_type_celltypist_per_cell"
] = preds_adata.obs[
    "predicted_labels"
].values
adata.obs[f"conf_score_celltypist"] = (
    preds_adata.obs["conf_score"].values
)

sc.pl.spatial(
    adata,
    color="cell_type_celltypist",
    spot_size=9,
    show=False,
)
plt.savefig(
    "cell_type_celltypist.png",
    dpi=2000,
    bbox_inches="tight",
)
plt.close()

# %%
