import anndata as ad
import novae
import scipy.sparse as sp
import scanpy as sc
from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
from components.spatial.qc import (
    filter_isolated_regions,
)


def preprocess_adatas(
    adatas,
    spatial_radius,
    slide_key,
    sample_list_names,
    use_highly_variable_genes=True,
    technology=None,
    coord_type="generic",
    run_dir=None,
    percentile_pct_mito=None,
    pct_mito=None,
    vmax=95,
):
    logger.info(
        f"Calculating spatial neighbors using Novae and {spatial_radius} with the slide key {slide_key}"
    )
    if (
        percentile_pct_mito is None
        and pct_mito is None
    ):
        raise ValueError(
            "Either percentile_pct_mito or pct_mito must be provided."
        )
    else:
        if (
            percentile_pct_mito
            is not None
        ):
            logger.info(
                f"Using percentile_pct_mito: {percentile_pct_mito}"
            )
        else:
            logger.info(
                f"Using pct_mito: {pct_mito}"
            )

    for i, data in enumerate(adatas):
        data.var["mt"] = (
            data.var_names.str.startswith(
                ("MT-", "mt-")
            )
        )
        logger.info(
            f"Data shape: {data}"
        )
        sample_name = data.obs[
            "sample_id"
        ][0]
        sc.pp.calculate_qc_metrics(
            data,
            inplace=True,
            percent_top=None,
        )
        sc.pp.calculate_qc_metrics(
            data,
            qc_vars=["mt"],
            inplace=True,
            percent_top=None,
        )
        fig = sc.pl.embedding(
            data,
            basis="spatial",
            color="n_counts",
            vmin=np.percentile(
                data.obs["n_counts"], 1
            ),
            vmax=np.percentile(
                data.obs["n_counts"],
                vmax,
            ),
            size=8,
            show=False,
            return_fig=True,
        )
        fig.savefig(
            f"{run_dir}/n_counts_embedding_{sample_name}.png",
            bbox_inches="tight",
            dpi=150,
        )
        plt.close(fig)

        fig = sc.pl.embedding(
            data,
            basis="spatial",
            color="pct_counts_mt",
            vmin=np.percentile(
                data.obs[
                    "pct_counts_mt"
                ],
                1,
            ),
            vmax=np.percentile(
                data.obs[
                    "pct_counts_mt"
                ],
                vmax,
            ),
            size=8,
            show=False,
            return_fig=True,
        )
        fig.savefig(
            f"{run_dir}/pct_counts_mt_embedding_{sample_name}.png",
            bbox_inches="tight",
            dpi=150,
        )
        plt.close(fig)

        if (
            percentile_pct_mito
            is not None
        ):
            pct_to_remove = data.obs[
                "pct_counts_mt"
            ].quantile(
                percentile_pct_mito
            )
        else:
            pct_to_remove = pct_mito

        adata_mt = data[
            data.obs["pct_counts_mt"]
            < pct_to_remove
        ]
        logger.info(
            f"Removed {data.n_obs - adata_mt.n_obs} cells with high mt percentage ({pct_to_remove}) for sample {sample_name}"
        )

        logger.info(
            f"Embedding plot after removing cells with mt percentage > than {pct_to_remove}"
        )
        fig = sc.pl.embedding(
            adata_mt,
            basis="spatial",
            color="pct_counts_mt",
            vmin=np.percentile(
                data.obs[
                    "pct_counts_mt"
                ],
                1,
            ),
            vmax=np.percentile(
                data.obs[
                    "pct_counts_mt"
                ],
                vmax,
            ),
            size=8,
            show=False,
            return_fig=True,  # This returns the figure object
        )
        fig.savefig(
            f"{run_dir}/mt_embedding_after_filtering_{sample_name}.png",
            bbox_inches="tight",
            dpi=150,
        )
        plt.close(fig)

        adatas[i] = adata_mt.copy()

    logger.info(
        "Computing initial spatial neighbors."
    )
    novae.spatial_neighbors(
        adatas,
        radius=spatial_radius,
        slide_key=slide_key,
        coord_type=coord_type,
        technology=technology,
    )

    logger.info(
        f"Filtering out isolated regions from adatas. \n {adatas}"
    )

    adatas = filter_isolated_regions(
        adatas,
        min_cells=2000,
        run_dir=run_dir,
        remove_specific_regions={
            "A1-GC-AHZT1": ["0"]
        },
    )

    logger.info(
        f"Re-computing spatial neighbors after filtering for isolated regions. \n {adatas}"
    )

    novae.spatial_neighbors(
        adatas,
        radius=spatial_radius,
        slide_key=slide_key,
        coord_type=coord_type,
        technology=technology,
    )

    logger.info(
        f"Neighbors calculated: \n {adatas}"
    )
    novae.plot.connectivities(adatas)
    logger.info(
        f"Saving connectivities plot to {run_dir}"
    )
    plt.savefig(
        f"{run_dir}/connectivities.png",
        dpi=600,
        bbox_inches="tight",
    )
    plt.close()

    logger.info(
        "Preprocessing adatas with Novae"
    )
    novae.utils.prepare_adatas(adatas)

    for data in adatas:
        data.layers[
            "lognorm_counts"
        ] = data.X.copy()

    logger.info("Concatenating adatas")
    adata_concat = ad.concat(
        adatas,
        label=slide_key,
        keys=sample_list_names,
        join="inner",
        merge="same",
    )

    # Rebuild spatial neighbors as block-diagonal (no cross-sample edges)
    adata_concat.obsp[
        "spatial_connectivities"
    ] = sp.block_diag(
        [
            adata.obsp[
                "spatial_connectivities"
            ]
            for adata in adatas
        ]
    ).tocsr()
    adata_concat.obsp[
        "spatial_distances"
    ] = sp.block_diag(
        [
            adata.obsp[
                "spatial_distances"
            ]
            for adata in adatas
        ]
    ).tocsr()

    logger.info("Running PCA")
    sc.pp.pca(
        adata_concat,
        use_highly_variable=use_highly_variable_genes,
    )

    logger.info(
        f"Returning both adatas \n {adatas} \n and concatenated adata...\n {adata_concat}"
    )
    return adatas, adata_concat
