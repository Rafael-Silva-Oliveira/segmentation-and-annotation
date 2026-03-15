# Reader for Visium HD Space Ranger outputs
# %%
import spatialdata as sd
import scanpy as sc
import pandas as pd
import numpy as np
import os
from matplotlib import rcParams
from matplotlib import font_manager
import matplotlib.pyplot as plt
import tifffile as tf
import sopa
from spatialdata import rasterize_bins
from typing import Iterable, Optional, Dict
import anndata as ad
import squidpy as sq
from loguru import logger
from components.spatial.pl import crop0,plot_cell_segmentation
from components.spatial.tl.config_loader import load_config
cfg = load_config(r"/mnt/work/RO_src/Projects/Paper_ST_and_scRNA/spatial/scripts/config.yaml")
sample_folders = cfg['sample_folders']
id_to_image = cfg['id_to_image']
sample_paths = cfg['sample_paths']
sample_mapping = cfg['sample_mapping']
outs = cfg['paths']['spaceranger_outs']

# %%
sdatas = []
for sample in sample_folders:
	sdata = sopa.io.visium_hd(
		f"{outs}/{sample}/outs",  # Space Ranger output directory
		fullres_image_file=id_to_image[sample], 
		dataset_id="",  # only if Space Ranger produced files with no prefix
	)
	sopa.make_image_patches(sdata)
	sopa.segmentation.stardist(sdata, min_area=20)
	sopa.segmentation.proseg(sdata, prior_shapes_key="auto")
	sopa.aggregate(sdata, aggregate_channels=False, expand_radius_ratio=1)
	# Update AnnData tables metadata and collect them
	for size, table in sdata.tables.items():
		table.var_names_make_unique()
		table.obs["annon_id"] = sample
		table.obs["sample_id"] = sample
		table.obs["sfh_id"] = sample_mapping[sample]
		# Add suffix identifier to table index
		table.obs.index = sample + "_" + table.obs.index.astype(str)
	sdata["square_002um"].X = sdata["square_002um"].X.tocsc()  # optimisation with the csc format
	lazy_bins_image = sd.rasterize_bins(
		sdata,
		bins="_square_002um",  # key of the bins shapes
		table_name="square_002um",  # key of the table with the bins gene expression
		row_key="array_row",
		col_key="array_col",
	)
	sdata["gene_expression_2_um"] = lazy_bins_image
	sdata.write(f"/mnt/archive/RO_src/data/spatial/processed/{sample}_binned_05012026.zarr", overwrite=True)
	sdatas.append(sdata)

concatenated_sdata = sd.concatenate( {sample_folders[0]:sdatas[0], sample_folders[1]:sdatas[1], sample_folders[2]:sdatas[2]},concatenate_tables=True)
concatenated_sdata.write(f"/mnt/archive/RO_src/data/spatial/processed/concatenated_binned_05012026.zarr", overwrite=True)