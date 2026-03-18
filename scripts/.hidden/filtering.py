import scanpy as sc
from loguru import logger
import matplotlib.pyplot as plt


def filter_spatial_anndata(
    concatenated_adata,
    sample_list_names,
    filter_cells_counts,
    filter_genes_counts,
    slide_key="sample_id",
):
    logger.info(
        f"Concatenated AnnData shape: {concatenated_adata.shape}"
    )

    n_genes_before = (
        concatenated_adata.shape[1]
    )

    sc.pp.filter_genes(
        concatenated_adata,
        min_counts=filter_genes_counts,
    )  # log1p(5) = 1.79 +-

    logger.info(
        f"Filtered genes: {n_genes_before} -> {concatenated_adata.shape[1]} ({n_genes_before - concatenated_adata.shape[1]} removed)"
    )

    # dict: sample_id -> AnnData (copied)
    adatas_by_sample = {
        sid: concatenated_adata[
            concatenated_adata.obs[
                slide_key
            ]
            == sid
        ].copy()
        for sid in sample_list_names
    }

    # list of AnnData objects (same order as sample_ids)
    adatas = [
        adatas_by_sample[sid]
        for sid in sample_list_names
    ]

    for i, adata in enumerate(adatas):
        logger.info(
            "Filtering cells..."
        )

        n_before = adata.n_obs

        sc.pp.filter_cells(
            adata,
            min_counts=filter_cells_counts,
        )  # adjust threshold as needed

        logger.info(
            f"{sample_list_names[i]}: {n_before} -> {adata.n_obs} bins ({n_before - adata.n_obs} removed)"
        )
        adatas[i] = adata

    return adatas


def filter_isolated_regions(
    adatas,
    min_cells=2000,
    run_dir=None,
    remove_specific_regions=None,
):
    from scipy.sparse.csgraph import (
        connected_components,
    )

    for i, adata in enumerate(adatas):
        coords = adata.obsm["spatial"]

        sample_id = adata.obs[
            "sample_id"
        ].unique()[0]

        logger.info(
            f"Running region filtering based on number of cells (removing isolated islands). {sample_id}"
        )

        print(
            f"X range: {coords[:, 0].min():.1f} - {coords[:, 0].max():.1f}"
        )
        print(
            f"Y range: {coords[:, 1].min():.1f} - {coords[:, 1].max():.1f}"
        )

        n_components, labels = (
            connected_components(
                adata.obsp[
                    "spatial_connectivities"
                ],
                directed=False,
            )
        )
        adata.obs["spatial_region"] = (
            labels.astype(str)
        )
        print(
            f"Found {n_components} spatial regions"
        )

        # Count cells per region
        region_counts = adata.obs[
            "spatial_region"
        ].value_counts()
        print(region_counts)

        large_regions = region_counts[
            region_counts >= min_cells
        ].index.tolist()
        print(
            f"Large regions (>={min_cells} cells): {large_regions}"
        )

        # Tag regions as "main" or "island"
        adata.obs["region_type"] = (
            adata.obs[
                "spatial_region"
            ].apply(
                lambda x: (
                    "main"
                    if x
                    in large_regions
                    else "island"
                )
            )
        )

        # Visualize the regions
        fig, axes = plt.subplots(
            1, 3, figsize=(14, 5)
        )

        # Plot all regions colored
        sc.pl.embedding(
            adata,
            basis="spatial",
            color="spatial_region",
            ax=axes[0],
            show=False,
            title="All spatial regions (based on connectivities)",
        )

        # Plot main vs island
        sc.pl.embedding(
            adata,
            basis="spatial",
            color="region_type",
            ax=axes[1],
            show=False,
            title="Main vs Islands",
        )

        if (
            remove_specific_regions
            is not None
        ):
            if (
                sample_id
                in remove_specific_regions
            ):
                regions_to_remove = remove_specific_regions[
                    sample_id
                ]
                adata = adata[
                    ~adata.obs[
                        "spatial_region"
                    ].isin(
                        regions_to_remove
                    )
                ]

                logger.info(
                    f"Removed user defined regions from {sample_id}: {regions_to_remove}"
                )
        cells_before_filtering = (
            adata.n_obs
        )
        # Filter to keep only main regions
        adata = adata[
            adata.obs["region_type"]
            == "main"
        ]
        cells_after_filtering = (
            adata.n_obs
        )
        print(
            f"Filtered: {cells_before_filtering} -> {cells_after_filtering} cells"
        )
        sc.pl.embedding(
            adata,
            basis="spatial",
            color="region_type",
            ax=axes[2],
            show=False,
            title="After filtering isolated islands",
        )
        plt.tight_layout()

        if run_dir is not None:
            plt.savefig(
                f"{run_dir}/spatial_regions_{sample_id}.png",
                dpi=300,
                bbox_inches="tight",
            )
        else:
            plt.savefig(
                f"spatial_regions_{sample_id}.png",
                dpi=300,
                bbox_inches="tight",
            )
        plt.show()

        adatas[i] = adata.copy()

    return adatas
