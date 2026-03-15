import spatialdata as sd
import spatialdata_io
import spatialdata_plot  # noqa: F401
from spatialdata import SpatialData
import scanpy as sc
import matplotlib.pyplot as plt


def crop0(x, crs, bbox):
    return sd.bounding_box_query(
        x,
        min_coordinate=[
            bbox["x"][0],
            bbox["y"][0],
        ],
        max_coordinate=[
            bbox["x"][1],
            bbox["y"][1],
        ],
        axes=("x", "y"),
        target_coordinate_system=crs,
    )


def plot_cell_segmentation(
    concatenated_sdata,
    boundaries_name="stardist_boundaries",
    coordinate_system="",
    grouping_obs="cell_id",
):
    image_elements = list(
        concatenated_sdata.images.keys()
    )
    image_elements_selected = [
        el
        for el in image_elements
        if "hires" in el
    ]
    shape_elements = list(
        concatenated_sdata.shapes.keys()
    )
    shape_elements_selected = [
        el
        for el in shape_elements
        if boundaries_name in el
    ]

    extents = []

    for i in range(
        len(image_elements_selected)
    ):
        extent = sd.get_extent(
            concatenated_sdata,
            elements=[
                shape_elements_selected[
                    i
                ]
            ],
            coordinate_system=coordinate_system,
        )
        extents.append(extent)

    # Fix all tables in concatenated_sdata
    for (
        table_name,
        adata,
    ) in concatenated_sdata.tables.items():
        if (
            "spatialdata_attrs"
            in adata.uns
        ):
            instance_key_col = (
                adata.uns[
                    "spatialdata_attrs"
                ].get(
                    "instance_key", None
                )
            )
            if (
                instance_key_col
                and instance_key_col
                in adata.obs.columns
            ):
                if (
                    str(
                        adata.obs[
                            instance_key_col
                        ].dtype
                    )
                    == "category"
                ):
                    print(
                        f"Fixing instance_key dtype in table '{table_name}'"
                    )
                    adata.obs[
                        instance_key_col
                    ] = adata.obs[
                        instance_key_col
                    ].astype(str)

    # Plotting
    if len(
        image_elements_selected
    ) != len(shape_elements_selected):
        print(
            "Check the spatial data to make sure that for every image there is a shape"
        )
    else:
        for i in range(
            len(image_elements_selected)
        ):
            print(
                "Plotting: "
                + image_elements_selected[
                    i
                ]
            )
            title = (
                image_elements_selected[
                    i
                ].replace(
                    "_hires_image", ""
                )
            )
            crop0(
                concatenated_sdata,
                crs="",
                bbox=extents[i],
            ).pl.render_images(
                image_elements_selected[
                    i
                ]
            ).pl.render_shapes(
                shape_elements_selected[
                    i
                ],
                color=grouping_obs,
            ).pl.show(
                coordinate_systems=coordinate_system,
                title=title,
            )


def plot_segmentation_overlap_qc(
    concatenated_sdata,
    sample_folders,
    bin_resolution="008um",
    segmentation_boundaries="stardist_boundaries",
    save_dir=None,
):
    import spatialdata as sd
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    import geopandas as gpd
    from shapely.strtree import STRtree

    fig, axes = plt.subplots(
        1,
        len(sample_folders),
        figsize=(
            5 * len(sample_folders),
            5,
        ),
    )

    for idx, sample_id in enumerate(
        sample_folders
    ):
        ax = (
            axes[idx]
            if len(sample_folders) > 1
            else axes
        )

        # Get bins and cell boundaries for this sample
        bin_key = f"_square_{bin_resolution}-{sample_id}"
        boundary_key = f"{segmentation_boundaries}-{sample_id}"

        bins_gdf = (
            concatenated_sdata.shapes[
                bin_key
            ]
        )
        cells_gdf = (
            concatenated_sdata.shapes[
                boundary_key
            ]
        )

        # Build spatial index for cell boundaries
        cell_tree = STRtree(
            cells_gdf.geometry.values
        )

        # Check which bins overlap with any cell
        overlaps = []
        for (
            bin_geom
        ) in bins_gdf.geometry:
            # Check if bin intersects any cell boundary
            intersecting_indices = cell_tree.query(
                bin_geom,
                predicate="intersects",
            )
            overlaps.append(
                len(
                    intersecting_indices
                )
                > 0
            )

        bins_gdf = bins_gdf.copy()
        bins_gdf["overlaps_cell"] = (
            overlaps
        )

        # Plot bins with overlap status
        # Non-overlapping bins in red, overlapping in green
        non_overlap = bins_gdf[
            ~bins_gdf["overlaps_cell"]
        ]
        overlap = bins_gdf[
            bins_gdf["overlaps_cell"]
        ]

        non_overlap.plot(
            ax=ax,
            color="red",
            alpha=0.5,
            edgecolor="none",
        )
        overlap.plot(
            ax=ax,
            color="green",
            alpha=0.5,
            edgecolor="none",
        )

        # Add statistics
        n_overlap = bins_gdf[
            "overlaps_cell"
        ].sum()
        n_total = len(bins_gdf)
        pct_overlap = (
            n_overlap / n_total
        ) * 100

        ax.set_title(
            f"{sample_id}\n{n_overlap:,}/{n_total:,} bins overlap ({pct_overlap:.1f}%)"
        )
        ax.set_aspect("equal")
        ax.axis("off")

    # Add legend
    legend_elements = [
        mpatches.Patch(
            facecolor="green",
            alpha=0.5,
            label="Overlaps segmented cells",
        ),
        mpatches.Patch(
            facecolor="red",
            alpha=0.5,
            label="No cell overlap",
        ),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.02),
    )

    plt.suptitle(
        f"Bin-Cell Segmentation Overlap QC ({bin_resolution} bins)",
        fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(
        f"{save_dir}/qc_bin_segmentation_overlap.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


def highlight_cell_types(
    adata,
    ct_to_highlight: list,
    run_dir: str,
    sample_names: list,
    cell_type_col: str = "cell_type",
):
    for sample_id in sample_names:
        adata_sub = adata[
            adata.obs["sample_id"]
            == sample_id
        ].copy()

        if ct_to_highlight == ["All"]:
            ct_to_highlight = (
                adata_sub.obs[
                    cell_type_col
                ].unique()
            )

        elif len(ct_to_highlight) >= 1:
            continue

        else:
            raise ValueError(
                "ct_to_highlight must be a list of cell types or ['All']"
            )

        for ct in ct_to_highlight:
            # Create a column that's either the cell type or "Other"
            adata_sub.obs[
                "highlight"
            ] = adata_sub.obs[ct].apply(
                lambda x: x
                if x == ct
                else "Other"
            )

            sc.pl.embedding(
                adata_sub,
                basis="spatial",
                color="highlight",
                palette={
                    ct: "red",
                    "Other": "lightgray",
                },
                show=False,
                alpha=0.5,
                size=8,
                title=f"{sample_id} - {ct}",
            )
            plt.savefig(
                f"/mnt/archive/RO_src/data/spatial/processed/{run_dir}/{sample_id}_{ct}_embedding.png",
                dpi=300,
                bbox_inches="tight",
            )

            plt.close()
