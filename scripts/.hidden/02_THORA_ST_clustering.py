# %%
# Standard library
import os
from datetime import (
    datetime,
)

# Third-party - data handling
from matplotlib.pylab import f
import numpy as np
import scipy.sparse as sp
import anndata as ad

# Third-party - visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import (
    rcParams,
)

# Third-party - single-cell & spatial analysis
import scanpy as sc
import scanpy.external as sce
import spatialdata as sd
import spatialleiden as sl
import novae

# Third-party - logging
from loguru import (
    logger,
)

# Local imports
from helper import (
    prepare_for_saving,
)

from components.spatial.qc import (
    filter_spatial_anndata,
    filter_isolated_regions,
)
from components.spatial.pp import (
    preprocess_adatas,
)
from components.spatial.pl import (
    crop0,
    plot_cell_segmentation,
    plot_segmentation_overlap_qc,
)
from components.spatial.tl.config_loader import (
    load_config,
)

# Matplotlib configuration
FIGSIZE = (
    3,
    3,
)
rcParams["figure.figsize"] = FIGSIZE
mpl.rcParams["figure.dpi"] = 300
plt.style.use("bmh")
plt.rcParams.update(
    {
        "figure.figsize": (
            12,
            8,
        ),
        "axes.facecolor": "white",
        "axes.edgecolor": "black",
    }
)

# Load configuration
cfg = load_config(
    r"/mnt/work/RO_src/Projects/Paper_ST_and_scRNA/spatial/scripts/config.yaml"
)
sample_folders = cfg["sample_folders"]
segmented_path = cfg["paths"][
    "segmented"
]
id_to_image = cfg["id_to_image"]
sample_paths = cfg["sample_paths"]
sample_mapping = cfg["sample_mapping"]
radius = cfg["params"]["radius"]
percentile_pct_mito = cfg["params"][
    "percentile_pct_mito"
]
pct_mito = cfg["params"]["pct_mito"]
filter_genes_counts = cfg["params"][
    "filter_genes_counts"
]
filter_cells_counts = cfg["params"][
    "filter_cells_counts"
]

# Setup run directory
run_name = (
    "novae_"
    + datetime.now().strftime(
        "%d%m%Y_%H%M"
    )
)
run_dir = f"{cfg['paths']['processed']}/{run_name}"
os.makedirs(
    run_dir,
    exist_ok=True,
)

logger.info(f"Run directory: {run_dir}")
logger.info(f"Running {run_name}")
logger.info(
    f"Current parameters are: {cfg}"
)

# logger.info(f"Plotting QC graph, showing segmented bins vs non segmented bins (i.e. bins that were not recognized or segmented as being actual cells)")
# plot_segmentation_overlap_qc(concatenated_sdata=concatenated_sdata, sample_folders=sample_folders, bin_resolution = "008um", segmentation_boundaries = "stardist_boundaries", save_dir = run_dir)

# %%
concatenated_sdata = sd.read_zarr(
    segmented_path
)
concat_adata = concatenated_sdata[
    "table"
].copy()

adatas = filter_spatial_anndata(
    concatenated_adata=concat_adata,
    sample_list_names=sample_folders,
    filter_cells_counts=filter_cells_counts,
    filter_genes_counts=filter_genes_counts,
    slide_key="sample_id",
)

# %%
adatas, adata_concat = (
    preprocess_adatas(
        adatas=adatas,
        spatial_radius=radius,
        slide_key="sample_id",
        sample_list_names=sample_folders,
        use_highly_variable_genes=True,
        run_dir=run_dir,
        percentile_pct_mito=percentile_pct_mito,
        pct_mito=pct_mito,
        vmax=99,
    )
)


# %%

logger.info(
    f"Overview of adatas after filtering regions with isolate islands:\n {adatas}"
)

# %%
# === NOVAE CLUSTERING ===
logger.info("Running Novae")
model = novae.Novae.from_pretrained(
    "MICS-Lab/novae-human-0"
)
model.fine_tune(adatas, max_epochs=50)
logger.info(
    f"Saving model on {run_dir}..."
)
model.save_pretrained(
    f"{run_dir}/model"
)

# Or comment code above and load a pretrained model instead
# pretrained_model_path = "/mnt/archive/RO_src/data/spatial/processed/novae_22012026_1329/model"
# logger.info(
#     f"Loading pretrained model from {pretrained_model_path}"
# )
# model = novae.Novae.from_pretrained(
#     pretrained_model_path
# )
model.compute_representations(adatas)

# %%
# Assign Novae domains
logger.info(
    "Assigning Novae domains..."
)
for domain in range(5, 41, 1):
    clustering_col = (
        f"novae_domains_{domain}"
    )
    model.assign_domains(
        adatas, level=domain
    )
    model.batch_effect_correction(
        adatas,
        obs_key=clustering_col,
    )

    for data in adatas:
        data.obsm[
            f"novae_latent_{domain}"
        ] = data.obsm[
            "novae_latent"
        ].copy()
        data.obsm[
            f"novae_latent_corrected_{domain}"
        ] = data.obsm[
            "novae_latent_corrected"
        ].copy()

    # Save plots
    novae.plot.domains(
        adatas,
        slide_name_key="sample_id",
        cell_size=8,
        show=False,
        obs_key=clustering_col,
    )
    plt.savefig(
        f"{run_dir}/domains_{domain}.png",
        dpi=1200,
        bbox_inches="tight",
    )
    plt.close()

    novae.plot.domains_proportions(
        adatas,
        obs_key=clustering_col,
        show=False,
    )
    plt.savefig(
        f"{run_dir}/domains_proportions_{domain}.png",
        dpi=600,
        bbox_inches="tight",
    )
    plt.close()


# %%
# === SAVE EVERYTHING ===
logger.info("Saving results...")

# Save individual adatas (contains both Novae + Spatial Leiden results)
for sample, data in zip(
    sample_folders, adatas
):
    data_save = prepare_for_saving(data)
    logger.info(
        f"Saving {sample} to {run_dir}/{sample}.h5ad..."
    )
    data_save.write_h5ad(
        f"{run_dir}/{sample}.h5ad"
    )

logger.info("Done!")
