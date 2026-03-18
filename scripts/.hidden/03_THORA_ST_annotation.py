# Reader for Visium HD Space Ranger outputs
# %%
from typing import Union

from helper import (
    prepare_for_saving,
    revert_from_conversion,
)
from components.spatial.tl.config_loader import (
    load_config,
)
from components.spatial.de import (
    rankby_group_scores,
)
from components.spatial.pl import (
    highlight_cell_types,
)
import json
import enrichmap as em

# import matplotlib.pyplot as plt
from loguru import logger

from sklearn.preprocessing import (
    MinMaxScaler,
)
import seaborn as sns
import scipy
import numpy as np
import pandas as pd
import scipy.stats as sts
from anndata import AnnData
import spatialdata_plot  # noqa: F401
import scanpy as sc
import os
from pathlib import Path
from matplotlib import rcParams
import matplotlib.pyplot as plt
import anndata as ad
import novae
import matplotlib as mpl
import squidpy as sq

rng = np.random.default_rng()


from sklearn.mixture import (
    GaussianMixture,
)
import numpy as np

# %%


class CellTypeAnnotator:
    """
    Annotates cell types using Enrichmap scoring.

    Results in column 'cell_type_enrichmap'.
    """

    def __init__(
        self,
        adata: AnnData,
        marker_genes: dict,
        clustering_col: str,
        output_dir: str,
        batch_key: str = "sample_id",
        domains_prop_filter: Union[
            float, int
        ] = 0.001,
        annotation_method: str = "gap",
        one_vs_all_test: str = "wilcoxon",
    ):
        self.adata = adata
        self.marker_genes = marker_genes
        self.clustering_col = (
            clustering_col
        )
        self.output_dir = Path(
            output_dir
        )
        self.batch_key = batch_key
        self.domains_prop_filter = (
            domains_prop_filter
        )
        self.output_dir.mkdir(
            parents=True, exist_ok=True
        )
        self.annotation_method = (
            annotation_method
        )
        self.one_vs_all_test = (
            one_vs_all_test
        )
        (
            self.output_dir / "figures"
        ).mkdir(exist_ok=True)
        # Filter markers to genes present in adata
        self.marker_genes = {
            k: [
                g
                for g in v
                if g in adata.var_names
            ]
            for k, v in marker_genes.items()
        }

        self.enrichmap_scores: (
            pd.DataFrame | None
        ) = None
        self.enrichmap_ranking: (
            pd.DataFrame | None
        ) = None

        # Storage for GMM diagnostic info (populated during gmm annotation)
        self._gmm_debug_info_enrichmap: dict = {}

        # Filter out domains with low proportion/count of cells
        if isinstance(
            self.domains_prop_filter,
            int,
        ):
            val_counts = self.adata.obs[
                self.clustering_col
            ].value_counts(
                normalize=False
            )
        else:
            val_counts = self.adata.obs[
                self.clustering_col
            ].value_counts(
                normalize=True
            )
        domains_to_keep = val_counts[
            val_counts
            > self.domains_prop_filter
        ].index
        n_before = self.adata.n_obs
        self.adata = self.adata[
            self.adata.obs[
                self.clustering_col
            ].isin(domains_to_keep)
        ].copy()

        logger.info(
            f"Filtered adata from {n_before} to {self.adata.n_obs} cells "
            f"(removed {len(val_counts) - len(domains_to_keep)} low-proportion domains)"
        )

    def run_enrichmap(
        self,
        smoothing: bool = True,
        correct_spatial_covariates: bool = True,
        save: bool = False,
    ) -> AnnData:
        """Run Enrichmap scoring."""
        logger.info(
            "Running Enrichmap scoring..."
        )

        em.tl.score(
            adata=self.adata,
            gene_set=self.marker_genes,
            smoothing=smoothing,
            correct_spatial_covariates=correct_spatial_covariates,
            batch_key=self.batch_key,
        )

        # Extract enrichmap scores
        score_cols = [
            col
            for col in self.adata.obs.columns
            if col.endswith("_score")
        ]
        self.enrichmap_scores = (
            self.adata.obs[
                score_cols
            ].copy()
        )

        if save:
            self._save_adata()

        return self.adata

    def _save_adata(self):
        """Save adata with all accumulated scores."""
        adata_to_save = (
            prepare_for_saving(
                self.adata.copy()
            )
        )
        adata_to_save.write_h5ad(
            self.output_dir
            / f"adata_scores_{self.clustering_col}.h5ad"
        )
        logger.info(
            f"Saved adata to {self.output_dir / f'adata_scores_{self.clustering_col}.h5ad'}"
        )

    def _save_enrichmap_plots(self):
        """Save gene_contributions_pca, morans_correlogram, and variogram_all plots per sample."""
        base_dir = (
            self.output_dir
            / "enrichmap_plots"
            / f"enrichmap_{self.clustering_col}"
        )
        # see if basedir exists if not create
        base_dir.mkdir(
            parents=True, exist_ok=True
        )

        score_cols = [
            col
            for col in self.adata.obs.columns
            if col.endswith("_score")
        ]
        if not score_cols:
            logger.warning(
                "No score columns found, skipping enrichmap plots."
            )
            return

        has_gene_contribs = (
            "gene_contributions"
            in self.adata.uns
        )

        # Gene contributions PCA per score
        if has_gene_contribs:
            for score_col in score_cols:
                score_name = (
                    score_col.replace(
                        "_score", ""
                    )
                )
                try:
                    em.pl.gene_contributions_pca(
                        self.adata,
                        score_key=score_col,
                        top_n_genes=5,
                        save=str(
                            base_dir
                            / f"gene_contributions_pca_{score_name}.png"
                        ),
                    )
                    plt.close("all")
                except Exception as e:
                    logger.warning(
                        f"gene_contributions_pca failed for {score_col}: {e}"
                    )
                    plt.close("all")
        else:
            logger.warning(
                "gene_contributions not in adata.uns, skipping PCA plots."
            )

        logger.info(
            f"Saved enrichmap plots to {base_dir}"
        )

    def _rank_groups_enrichmap(
        self,
        sort_by=["group", "meanchange"],
    ) -> pd.DataFrame:
        """Rank enrichmap scores by group."""
        logger.info(
            f"Using {self.one_vs_all_test} test for Enrichmap"
        )

        logger.info(
            f"Sorting by {sort_by}"
        )
        logger.info(
            f"Adata is: {self.adata}"
        )
        df = rankby_group_scores(
            adata=self.adata,
            groupby=self.clustering_col,
            reference="rest",
            method=self.one_vs_all_test,
        )
        # df = df[df["padj"] <= 0.05]
        # df = df[df["meanchange"] > 0]
        df = df.sort_values(
            by=sort_by,
            ascending=[True, False],
        )

        return df

    def get_top_celltypes_per_domain(
        self,
        n_top: int | str = 6,
        method: str = "enrichmap",
    ) -> dict:

        if method == "enrichmap":
            if (
                self.enrichmap_scores
                is None
            ):
                raise ValueError(
                    "Run run_enrichmap() first"
                )
            df = self._rank_groups_enrichmap()
            self.enrichmap_ranking = df
            df.to_csv(
                self.output_dir
                / f"enrichment_ranking_{self.clustering_col}.csv",
                index=True,
            )
        else:
            raise ValueError(
                f"Unknown method '{method}'"
            )

        if isinstance(n_top, int):
            ctypes_dict = (
                df.groupby("group")
                .head(n_top)
                .groupby("group")[
                    "name"
                ]
                .apply(
                    lambda x: list(x)
                )
                .to_dict()
            )

        elif isinstance(n_top, str):
            ctypes_dict = (
                df.groupby("group")[
                    "name"
                ]
                .apply(
                    lambda x: list(x)
                )
                .to_dict()
            )
        else:
            raise ValueError(
                "n_top must be an integer or 'all'"
            )

        return ctypes_dict

    def assign_cell_types(
        self, n_top: int | str = 6
    ) -> AnnData:
        """
        Assign cell types using Enrichmap scores.

        Creates column 'cell_type_enrichmap'.
        """

        if (
            self.annotation_method
            not in [
                "gap",
                "gmm",
            ]
        ):
            raise ValueError(
                "annotation_method must be 'gap', or 'gmm'"
            )

        logger.info(
            f"Using annotation method: {self.annotation_method}"
        )

        if (
            self.enrichmap_scores
            is None
        ):
            raise ValueError(
                "Enrichmap has not been run. Please run run_enrichmap() first."
            )

        self._assign_enrichmap(
            n_top=n_top,
            annotation_method=self.annotation_method,
        )

        return self.adata

    def _apply_gmm_with_gap(
        self,
        domain_data: pd.DataFrame,
    ) -> tuple[str, dict]:
        """
        Pre-filter to positive meanchange, fit 1 vs 2 component GMM (BIC),
        then apply gap method within the enriched component.

        Returns (annotation_string, debug_dict) where debug_dict contains
        all info needed for diagnostic plotting.
        """
        domain_data = domain_data[
            (
                domain_data[
                    "meanchange"
                ]
                > 0
            )
            & (
                domain_data["padj"]
                < 0.05
            )
        ].copy()

        debug: dict = {
            "domain_data": domain_data,
            "gmm2": None,
            "labels": None,
            "enriched_comp": None,
            "selected": [],
            "is_bimodal": False,
        }

        if len(domain_data) == 0:
            return "Unknown", debug

        # For early returns, store domain_data as enriched_data
        # so the elbow plot can show the full curve
        if len(domain_data) <= 2:
            debug["enriched_data"] = (
                domain_data
            )
            debug["selected"] = [
                domain_data[
                    "name"
                ].iloc[0]
            ]
            return (
                domain_data[
                    "name"
                ].iloc[0],
                debug,
            )

        X = domain_data[
            "meanchange"
        ].values.reshape(-1, 1)

        gmm1 = GaussianMixture(
            n_components=1,
            random_state=42,
        ).fit(X)
        gmm2 = GaussianMixture(
            n_components=2,
            random_state=42,
        ).fit(X)

        if gmm1.bic(X) <= gmm2.bic(X):
            # Unimodal: single top type
            debug["enriched_data"] = (
                domain_data
            )
            debug["selected"] = [
                domain_data[
                    "name"
                ].iloc[0]
            ]
            return (
                domain_data[
                    "name"
                ].iloc[0],
                debug,
            )

        debug["is_bimodal"] = True
        debug["gmm2"] = gmm2
        enriched_comp = int(
            np.argmax(
                gmm2.means_.ravel()
            )
        )
        labels = gmm2.predict(X)
        debug["labels"] = labels
        debug["enriched_comp"] = (
            enriched_comp
        )

        enriched_mask = (
            labels == enriched_comp
        )
        enriched_data = domain_data[
            enriched_mask
        ].copy()

        debug["enriched_data"] = (
            enriched_data
        )

        if len(enriched_data) == 0:
            debug["selected"] = [
                domain_data[
                    "name"
                ].iloc[0]
            ]
            return (
                domain_data[
                    "name"
                ].iloc[0],
                debug,
            )

        if len(enriched_data) == 1:
            debug["selected"] = (
                enriched_data[
                    "name"
                ].tolist()
            )
            return (
                enriched_data[
                    "name"
                ].iloc[0],
                debug,
            )

        # Gap method within enriched component (already sorted desc)
        sorted_mc = enriched_data[
            "meanchange"
        ].values
        gaps = (
            sorted_mc[:-1]
            - sorted_mc[1:]
        )
        biggest_gap_idx = int(
            np.argmax(gaps)
        )
        selected_names = (
            enriched_data["name"]
            .iloc[: biggest_gap_idx + 1]
            .tolist()
        )

        debug["enriched_data"] = (
            enriched_data
        )
        debug["biggest_gap_idx"] = (
            biggest_gap_idx
        )
        debug["selected"] = (
            selected_names
        )
        return "/".join(
            selected_names
        ), debug

    def _assign_enrichmap(
        self,
        n_top: int | str,
        annotation_method: str,
    ):
        """Assign cell types using Enrichmap scores."""

        if annotation_method == "gap":
            _ = self.get_top_celltypes_per_domain(
                n_top="all",
                method="enrichmap",
            )
            logger.info(
                "Applying gap-based detection on meanchange (1-vs-all enrichment scores)"
            )

            # Re-order the domains by their meanchange
            assert (
                self.enrichmap_ranking
                is not None
            )
            enrichmap_ranks = self.enrichmap_ranking.groupby(
                "group",
                group_keys=False,
            ).apply(
                lambda x: x.sort_values(
                    "meanchange",
                    ascending=False,
                )
            )

            enrichmap_gap_annotations = {}
            for domain in enrichmap_ranks.group.unique():
                domain_data = enrichmap_ranks.loc[
                    enrichmap_ranks.group
                    == domain
                ].copy()

                # Only consider significant cell types with positive meanchange
                logger.info(
                    "Filtering by padj < 0.05 and meanchange > 0"
                )
                domain_data = domain_data.loc[
                    (
                        domain_data[
                            "meanchange"
                        ]
                        > 0
                    )
                    & (
                        domain_data[
                            "padj"
                        ]
                        < 0.05
                    )
                ]

                if (
                    len(domain_data)
                    == 0
                ):
                    enrichmap_gap_annotations[
                        domain
                    ] = "Unknown"
                    continue

                if (
                    len(domain_data)
                    == 1
                ):
                    enrichmap_gap_annotations[
                        domain
                    ] = domain_data[
                        "name"
                    ].iloc[0]
                    continue

                sorted_mc = domain_data[
                    "meanchange"
                ].values

                # Find the largest drop between consecutive values
                gaps = (
                    sorted_mc[:-1]
                    - sorted_mc[1:]
                )
                biggest_gap_idx = (
                    np.argmax(gaps)
                )

                # Take everything above the biggest gap
                enrichmap_gap_annotations[
                    domain
                ] = "/".join(
                    domain_data["name"]
                    .iloc[
                        : biggest_gap_idx
                        + 1
                    ]
                    .tolist()
                )

            logger.info(
                f"Mapping the following domains to the cell types: \n {enrichmap_gap_annotations}"
            )
            self.adata.obs[
                "cell_type_enrichmap"
            ] = self.adata.obs[
                self.clustering_col
            ].map(
                lambda x: (
                    enrichmap_gap_annotations.get(
                        x, "Unknown"
                    )
                )
            )

        elif annotation_method == "gmm":
            _ = self.get_top_celltypes_per_domain(
                n_top="all",
                method="enrichmap",
            )
            logger.info(
                "Applying GMM-based detection on meanchange (1-vs-all enrichment scores)"
            )

            assert (
                self.enrichmap_ranking
                is not None
            )
            enrichmap_ranks = self.enrichmap_ranking.groupby(
                "group",
                group_keys=False,
            ).apply(
                lambda x: x.sort_values(
                    "meanchange",
                    ascending=False,
                )
            )

            gmm_annotations = {}
            self._gmm_debug_info_enrichmap = {}
            for domain in enrichmap_ranks.group.unique():
                domain_data = enrichmap_ranks.loc[
                    enrichmap_ranks.group
                    == domain
                ].copy()

                annotation, debug = (
                    self._apply_gmm_with_gap(
                        domain_data
                    )
                )
                gmm_annotations[
                    domain
                ] = annotation
                self._gmm_debug_info_enrichmap[
                    domain
                ] = debug

            logger.info(
                f"Mapping the following domains to the cell types: \n {gmm_annotations}"
            )
            self.adata.obs[
                "cell_type_enrichmap"
            ] = self.adata.obs[
                self.clustering_col
            ].map(
                lambda x: (
                    gmm_annotations.get(
                        x, "Unknown"
                    )
                )
            )

        # Per-spot high granularity: argmax over only the significant
        # cell types (meanchange > 0, padj < 0.05) for each domain
        if (
            self.enrichmap_scores
            is not None
            and self.enrichmap_ranking
            is not None
        ):
            hg_col = pd.Series(
                "Unknown",
                index=self.adata.obs_names,
            )

            sig_ranking = self.enrichmap_ranking[
                (
                    self.enrichmap_ranking[
                        "meanchange"
                    ]
                    > 0
                )
                & (
                    self.enrichmap_ranking[
                        "padj"
                    ]
                    < 0.05
                )
            ]

            for (
                domain,
                grp,
            ) in sig_ranking.groupby(
                "group"
            ):
                domain_mask = (
                    self.adata.obs[
                        self.clustering_col
                    ]
                    == domain
                )
                if (
                    domain_mask.sum()
                    == 0
                ):
                    continue

                sig_score_cols = [
                    f"{ct}_score"
                    for ct in grp[
                        "name"
                    ].values
                    if f"{ct}_score"
                    in self.enrichmap_scores.columns
                ]
                if not sig_score_cols:
                    continue

                domain_scores = self.enrichmap_scores.loc[
                    domain_mask,
                    sig_score_cols,
                ]
                hg_col.loc[
                    domain_mask
                ] = domain_scores.idxmax(
                    axis=1
                ).str.replace(
                    "_score", ""
                )

            self.adata.obs[
                "cell_type_enrichmap_high_granularity"
            ] = hg_col
            logger.info(
                "Assigned 'cell_type_enrichmap_high_granularity' "
                "(per-spot argmax over significant cell types per domain)"
            )

        logger.info(
            "Assigned 'cell_type_enrichmap'"
        )

    def plot_matrixplot(
        self,
        method: str = "enrichmap",
        n_top: int = 6,
        save: bool = True,
        **kwargs,
    ):
        """Plot matrix plot of scores per domain."""
        ctypes_dict = self.get_top_celltypes_per_domain(
            n_top=n_top, method=method
        )

        score_cols = [
            col
            for col in self.adata.obs.columns
            if col.endswith("_score")
        ]
        # Create AnnData with scores as variables
        score_matrix = self.adata.obs[
            score_cols
        ].values
        score_adata = AnnData(
            X=score_matrix,
            obs=self.adata.obs[
                [self.clustering_col]
            ].copy(),
        )
        score_adata.var_names = [
            col.replace("_score", "")
            for col in score_cols
        ]

        sc.tl.dendrogram(
            self.adata,
            groupby=self.clustering_col,
        )

        sc.pl.matrixplot(
            adata=score_adata,
            var_names=ctypes_dict,
            groupby=self.clustering_col,
            dendrogram=True,
            standard_scale="var",
            colorbar_title="Z-scaled scores",
            cmap="RdBu_r",
            swap_axes=True,
            show=False,
            **kwargs,
        )

        if save:
            plt.savefig(
                self.output_dir
                / "figures"
                / f"matrixplot_{method}_{self.clustering_col}.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()
            logger.info(
                f"Saved matrixplot to {self.output_dir / 'figures' / f'matrixplot_{method}_{self.clustering_col}.png'}"
            )

    def plot_spatial(
        self,
        color_col: str = "cell_type_enrichmap",
        save: bool = True,
        **kwargs,
    ):
        """Plot spatial distribution of cell types per sample."""
        for sample_id in self.adata.obs[
            self.batch_key
        ].unique():
            adata_sub = self.adata[
                self.adata.obs[
                    self.batch_key
                ]
                == sample_id
            ].copy()

            logger.info(
                f"Plotting {color_col} spatial distribution for {sample_id}"
            )
            sc.pl.spatial(
                adata_sub,
                color=color_col,
                show=False,
                **kwargs,
            )

            if save:
                plt.savefig(
                    self.output_dir
                    / "figures"
                    / f"spatial_{sample_id}_{color_col}_{self.clustering_col}.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close()

    def plot_violin(
        self,
        method: str = "enrichmap",
        save: bool = True,
        **kwargs,
    ):

        if method == "enrichmap":
            if (
                self.enrichmap_scores
                is None
            ):
                raise ValueError(
                    "Run run_enrichmap() first"
                )
        else:
            raise ValueError(
                f"Unknown method '{method}'"
            )

        for sample_id in self.adata.obs[
            self.batch_key
        ].unique():
            adata_sample = self.adata[
                self.adata.obs[
                    self.batch_key
                ]
                == sample_id
            ].copy()

            # Enrichmap scores are in obs columns
            score_cols = [
                col
                for col in adata_sample.obs.columns
                if col.endswith(
                    "_score"
                )
            ]
            # Create AnnData with scores as X
            score_matrix = (
                adata_sample.obs[
                    score_cols
                ].values
            )
            score_adata = AnnData(
                X=score_matrix,
                obs=adata_sample.obs[
                    [
                        self.clustering_col
                    ]
                ].copy(),
            )
            score_adata.var_names = [
                col.replace(
                    "_score", ""
                )
                for col in score_cols
            ]
            score_cols = score_adata.var_names.tolist()

            for cell_type in score_cols:
                logger.info(
                    f"Plotting {method} violin for {cell_type} in {sample_id}"
                )
                sc.pl.violin(
                    score_adata,
                    keys=[cell_type],
                    groupby=self.clustering_col,
                    rotation=90,
                    show=False,
                    **kwargs,
                )

                if save:
                    plt.savefig(
                        self.output_dir
                        / "figures"
                        / f"violin_{sample_id}_{cell_type}_{method}_{self.clustering_col}.png",
                        dpi=300,
                        bbox_inches="tight",
                    )
                    plt.close()

    def _annotate_scatter_points(
        self,
        ax,
        mc_values: np.ndarray,
        names: np.ndarray,
        point_colors: list,
        markers: list,
        sizes: list,
        x_span: float,
    ) -> None:
        """
        Scatter cell types along y=0 and annotate with staggered y offsets to
        avoid label overlap.  Points that are within `min_gap` of their
        x-neighbour are pushed to a deeper level.  Deeper labels get a thin
        connector arrow drawn back to the point.
        """
        n = len(mc_values)
        if n == 0:
            return

        # Assign stagger levels sorted by x
        sort_order = np.argsort(
            mc_values
        )
        min_gap = x_span * 0.06
        y_levels = np.zeros(
            n, dtype=int
        )
        level = 0
        for rank in range(1, n):
            i = sort_order[rank]
            prev_i = sort_order[
                rank - 1
            ]
            if (
                mc_values[i]
                - mc_values[prev_i]
            ) < min_gap:
                level += 1
            else:
                level = 0
            y_levels[i] = level

        # px below y=0 per level
        y_offsets_px = [
            -14,
            -28,
            -42,
            -56,
        ]

        for i in range(n):
            mc = mc_values[i]
            name = names[i]
            color = point_colors[i]
            marker = markers[i]
            size = sizes[i]
            lvl = min(
                y_levels[i],
                len(y_offsets_px) - 1,
            )
            y_off = y_offsets_px[lvl]

            ax.scatter(
                mc,
                0,
                color=color,
                marker=marker,
                s=size,
                zorder=5,
                clip_on=False,
            )
            arrow_props = (
                dict(
                    arrowstyle="-",
                    color="gray",
                    lw=0.4,
                    shrinkA=0,
                    shrinkB=3,
                )
                if lvl > 0
                else None
            )
            ax.annotate(
                name,
                (mc, 0),
                textcoords="offset points",
                xytext=(0, y_off),
                fontsize=4.5,
                ha="center",
                rotation=60,
                arrowprops=arrow_props,
            )

    def plot_gmm_diagnostics(
        self,
        method: str = "enrichmap",
        save: bool = True,
    ) -> None:
        """
        Diagnostic plot showing GMM component fits per domain.

        For each domain shows:
        - The meanchange distribution (filtered to positive values)
        - The two fitted Gaussian components with shading
        - The GMM decision boundary (dashed vertical line)
        - Each cell type as a point, coloured by component
        - Stars marking which cell types were finally selected
          after gap refinement within the enriched component

        Must be called after assign_cell_types() with annotation_method='gmm'.
        """
        from scipy.stats import (
            norm as sp_norm,
        )

        if method == "enrichmap":
            debug_info = self._gmm_debug_info_enrichmap
        else:
            raise ValueError(
                f"Unknown method '{method}'"
            )

        if not debug_info:
            raise ValueError(
                f"No GMM debug info for '{method}'. "
                "Run assign_cell_types() with annotation_method='gmm' first."
            )

        domains = list(
            debug_info.keys()
        )
        n_domains = len(domains)
        n_cols = min(4, n_domains)
        n_rows = int(
            np.ceil(n_domains / n_cols)
        )

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(
                5 * n_cols,
                4 * n_rows,
            ),
        )
        axes_flat = np.array(
            axes
        ).flatten()

        for ax_idx, domain in enumerate(
            domains
        ):
            ax = axes_flat[ax_idx]
            info = debug_info[domain]
            domain_data = info[
                "domain_data"
            ]
            selected = set(
                info["selected"]
            )

            ax.set_title(
                f"Domain {domain}",
                fontsize=8,
            )
            ax.set_xlabel(
                "meanchange", fontsize=7
            )
            ax.tick_params(labelsize=6)

            if len(domain_data) == 0:
                ax.text(
                    0.5,
                    0.5,
                    "Unknown\n(no positive enrichment)",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=7,
                )
                continue

            mc_values = domain_data[
                "meanchange"
            ].values
            names = domain_data[
                "name"
            ].values
            x_min = (
                mc_values.min() - 0.05
            )
            x_max = (
                mc_values.max() + 0.05
            )
            x_range = np.linspace(
                x_min, x_max, 300
            )

            if (
                info["is_bimodal"]
                and info["gmm2"]
                is not None
            ):
                gmm2 = info["gmm2"]
                labels = info["labels"]
                enriched_comp = info[
                    "enriched_comp"
                ]

                # Draw each Gaussian component
                for comp_idx in range(
                    2
                ):
                    mean = gmm2.means_[
                        comp_idx, 0
                    ]
                    std = np.sqrt(
                        gmm2.covariances_[
                            comp_idx,
                            0,
                            0,
                        ]
                    )
                    weight = (
                        gmm2.weights_[
                            comp_idx
                        ]
                    )
                    is_enriched_comp = (
                        comp_idx
                        == enriched_comp
                    )
                    color = (
                        "steelblue"
                        if is_enriched_comp
                        else "silver"
                    )
                    label = (
                        "enriched"
                        if is_enriched_comp
                        else "background"
                    )
                    y_vals = (
                        weight
                        * sp_norm.pdf(
                            x_range,
                            mean,
                            std,
                        )
                    )
                    ax.plot(
                        x_range,
                        y_vals,
                        color=color,
                        lw=2,
                        label=label,
                    )
                    ax.fill_between(
                        x_range,
                        y_vals,
                        alpha=0.2,
                        color=color,
                    )

                # Decision boundary where the predicted component changes
                preds = gmm2.predict(
                    x_range.reshape(
                        -1, 1
                    )
                )
                transitions = np.where(
                    np.diff(preds)
                )[0]
                for t in transitions:
                    ax.axvline(
                        x_range[t],
                        color="black",
                        ls="--",
                        lw=1,
                        alpha=0.7,
                    )

                # Scatter cell types along y=0 with staggered labels
                pt_colors = [
                    "steelblue"
                    if labels[i]
                    == enriched_comp
                    else "silver"
                    for i in range(
                        len(mc_values)
                    )
                ]
                pt_markers = [
                    "*"
                    if names[i]
                    in selected
                    else "o"
                    for i in range(
                        len(mc_values)
                    )
                ]
                pt_sizes = [
                    120
                    if names[i]
                    in selected
                    else 40
                    for i in range(
                        len(mc_values)
                    )
                ]
                self._annotate_scatter_points(
                    ax,
                    mc_values,
                    names,
                    pt_colors,
                    pt_markers,
                    pt_sizes,
                    x_span=x_max
                    - x_min,
                )

                ax.legend(
                    fontsize=5,
                    loc="upper left",
                )

            else:
                # Unimodal or too few points
                pt_colors = [
                    "steelblue"
                    if n in selected
                    else "silver"
                    for n in names
                ]
                self._annotate_scatter_points(
                    ax,
                    mc_values,
                    names,
                    pt_colors,
                    ["o"] * len(names),
                    [50] * len(names),
                    x_span=x_max
                    - x_min,
                )
                ax.text(
                    0.5,
                    0.7,
                    "unimodal",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=7,
                    color="gray",
                )

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(bottom=0)
            ax.set_ylabel(
                "density", fontsize=7
            )

        for ax_idx in range(
            n_domains, len(axes_flat)
        ):
            axes_flat[ax_idx].axis(
                "off"
            )

        fig.suptitle(
            f"GMM diagnostics — {method} — {self.clustering_col}",
            fontsize=11,
        )
        plt.tight_layout()

        if save:
            path = (
                self.output_dir
                / "figures"
                / f"gmm_diagnostics_{method}_{self.clustering_col}.png"
            )
            plt.savefig(
                path,
                dpi=200,
                bbox_inches="tight",
            )
            plt.close()
            logger.info(
                f"Saved GMM diagnostics to {path}"
            )
        else:
            plt.show()

    def plot_gmm_elbow(
        self,
        method: str = "enrichmap",
        save: bool = True,
    ) -> None:
        """
        Elbow / scree-style plot per domain showing meanchange (y) vs
        ranked cell types (x) within the GMM enriched component.

        Highlights the gap position with a vertical dashed line and
        shades the selected cell types.
        Must be called after assign_cell_types() with annotation_method='gmm'.
        """
        if method == "enrichmap":
            debug_info = self._gmm_debug_info_enrichmap
        else:
            raise ValueError(
                f"Unknown method '{method}'"
            )

        if not debug_info:
            raise ValueError(
                f"No GMM debug info for '{method}'. "
                "Run assign_cell_types() with "
                "annotation_method='gmm' first."
            )

        # Plot domains that have enriched data
        plottable = {
            d: info
            for d, info in debug_info.items()
            if info.get("enriched_data")
            is not None
            and len(
                info.get(
                    "enriched_data", []
                )
            )
            > 0
        }

        if not plottable:
            logger.warning(
                "No domains with enriched data to plot."
            )
            return

        domains = list(plottable.keys())
        n_domains = len(domains)
        n_cols = min(4, n_domains)
        n_rows = int(
            np.ceil(n_domains / n_cols)
        )

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(
                5 * n_cols,
                4 * n_rows,
            ),
        )
        axes_flat = np.array(
            axes
        ).flatten()

        for ax_idx, domain in enumerate(
            domains
        ):
            ax = axes_flat[ax_idx]
            info = plottable[domain]
            domain_data = info[
                "domain_data"
            ]
            enriched_data = info[
                "enriched_data"
            ]
            gap_idx = info.get(
                "biggest_gap_idx"
            )
            selected = set(
                info["selected"]
            )
            # Use all positive-meanchange cell types
            mc_vals = domain_data[
                "meanchange"
            ].values
            ct_names = domain_data[
                "name"
            ].values
            x_pos = np.arange(
                len(mc_vals)
            )

            # Determine enriched names for colouring
            enriched_names = set(
                enriched_data[
                    "name"
                ].values
            )

            # Line connecting all points
            ax.plot(
                x_pos,
                mc_vals,
                color="gray",
                lw=1.5,
                zorder=2,
            )

            # Colour points: selected (blue) / enriched-not-selected (lightblue) / background (silver)
            for i, (
                xp,
                mc,
                name,
            ) in enumerate(
                zip(
                    x_pos,
                    mc_vals,
                    ct_names,
                )
            ):
                is_sel = (
                    name in selected
                )
                is_enriched = (
                    name
                    in enriched_names
                )
                if is_sel:
                    color = "steelblue"
                    ec = "black"
                elif is_enriched:
                    color = (
                        "lightskyblue"
                    )
                    ec = "steelblue"
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

            # GMM boundary: vertical line between last enriched and first background
            n_enriched = len(
                enriched_names
            )
            if (
                n_enriched > 0
                and n_enriched
                < len(mc_vals)
            ):
                gmm_x = n_enriched - 0.5
                ax.axvline(
                    gmm_x,
                    color="gray",
                    ls=":",
                    lw=1,
                    alpha=0.7,
                    label="GMM split",
                )

            # Gap cutoff within enriched component
            if gap_idx is not None:
                gap_x = gap_idx + 0.5
                gap_val = (
                    enriched_data[
                        "meanchange"
                    ].values[gap_idx]
                    - enriched_data[
                        "meanchange"
                    ].values[
                        gap_idx + 1
                    ]
                )
                ax.axvline(
                    gap_x,
                    color="red",
                    ls="--",
                    lw=1,
                    alpha=0.8,
                    label=f"gap = {gap_val:.3f}",
                )

                # Shade selected region
                ax.axvspan(
                    -0.5,
                    gap_x,
                    alpha=0.08,
                    color="steelblue",
                )

                # Bracket showing the gap magnitude
                y_top = enriched_data[
                    "meanchange"
                ].values[gap_idx]
                y_bot = enriched_data[
                    "meanchange"
                ].values[gap_idx + 1]
                bracket_x = gap_x + 0.3
                ax.annotate(
                    "",
                    xy=(
                        bracket_x,
                        y_bot,
                    ),
                    xytext=(
                        bracket_x,
                        y_top,
                    ),
                    arrowprops=dict(
                        arrowstyle="<->",
                        color="red",
                        lw=1.2,
                    ),
                )
                ax.text(
                    bracket_x + 0.15,
                    (y_top + y_bot) / 2,
                    f"{y_top - y_bot:.3f}",
                    fontsize=6,
                    color="red",
                    va="center",
                )

            # X-axis: cell type names
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
                f"Domain {domain}",
                fontsize=8,
            )
            ax.legend(
                fontsize=5,
                loc="upper right",
            )
            ax.tick_params(labelsize=6)

        for ax_idx in range(
            n_domains, len(axes_flat)
        ):
            axes_flat[ax_idx].axis(
                "off"
            )

        fig.suptitle(
            f"GMM+gap elbow — {method} — {self.clustering_col}",
            fontsize=11,
        )
        plt.tight_layout()

        if save:
            path = (
                self.output_dir
                / "figures"
                / f"gmm_elbow_{method}_{self.clustering_col}.png"
            )
            plt.savefig(
                path,
                dpi=400,
                bbox_inches="tight",
            )
            plt.close()
            logger.info(
                f"Saved GMM elbow plot to {path}"
            )
        else:
            plt.show()

    def _get_score_cols_for_domain(
        self,
        d_adata,
        sig_celltypes,
        method: str,
    ) -> list[str]:
        """
        Get score column names for significant cell types
        and ensure they exist in d_adata.obs.

        Returns the list of score column names present in
        d_adata.obs after this call.
        """
        score_cols = []

        score_cols = [
            f"{ct}_score"
            for ct in sig_celltypes
            if f"{ct}_score"
            in d_adata.obs.columns
        ]
        if not score_cols:
            if (
                self.enrichmap_scores
                is not None
            ):
                avail = self.enrichmap_scores.columns
                score_cols = [
                    c
                    for c in avail
                    if c.replace(
                        "_score", ""
                    )
                    in sig_celltypes
                ]
                for col in score_cols:
                    d_adata.obs[col] = (
                        self.enrichmap_scores.loc[
                            d_adata.obs_names,
                            col,
                        ].values
                    )

        return score_cols

    def plot_gmm_morans(
        self,
        method: str = "enrichmap",
        n_neighs: int = 20,
        n_perms: int = 400,
        save: bool = True,
    ) -> None:
        """
        Moran's I spatial autocorrelation bar plot per domain,
        computed separately for each sample_id to avoid
        spurious cross-sample spatial neighbors.

        Uses only cell type scores whose rankings have
        meanchange > 0 and padj < 0.05 (same filter as the
        elbow plot).

        Produces one figure per sample, each with a grid of
        subplots (one per domain) matching the layout of the
        other diagnostic plots.

        Must be called after assign_cell_types() with
        annotation_method='gmm'.
        """
        if method == "enrichmap":
            debug_info = self._gmm_debug_info_enrichmap
        else:
            raise ValueError(
                f"Unknown method '{method}'"
            )

        if not debug_info:
            raise ValueError(
                f"No GMM debug info for '{method}'. "
                "Run assign_cell_types() with "
                "annotation_method='gmm' first."
            )

        # Plot domains that have significant cell types
        # (domain_data is already filtered to
        # meanchange > 0 and padj < 0.05)
        plottable = {
            d: info
            for d, info in debug_info.items()
            if info.get("domain_data")
            is not None
            and len(
                info.get(
                    "domain_data", []
                )
            )
            > 0
        }

        if not plottable:
            logger.warning(
                "No domains with significant data to plot Moran's I."
            )
            return

        domains = list(plottable.keys())
        n_domains = len(domains)
        samples = (
            self.adata.obs[
                self.batch_key
            ]
            .unique()
            .tolist()
        )

        for sample_id in samples:
            sample_mask = (
                self.adata.obs[
                    self.batch_key
                ]
                == sample_id
            )

            n_cols = min(4, n_domains)
            n_rows = int(
                np.ceil(
                    n_domains / n_cols
                )
            )

            fig, axes = plt.subplots(
                n_rows,
                n_cols,
                figsize=(
                    5 * n_cols,
                    4 * n_rows,
                ),
            )
            axes_flat = np.array(
                axes
            ).flatten()

            for (
                ax_idx,
                domain,
            ) in enumerate(domains):
                ax = axes_flat[ax_idx]
                info = plottable[domain]
                domain_data = info[
                    "domain_data"
                ]
                selected = set(
                    info["selected"]
                )

                sig_celltypes = (
                    domain_data[
                        "name"
                    ].values
                )

                if (
                    len(sig_celltypes)
                    == 0
                ):
                    ax.text(
                        0.5,
                        0.5,
                        "No significant\ncell types",
                        transform=ax.transAxes,
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="gray",
                    )
                    ax.set_title(
                        f"Domain {domain}",
                        fontsize=8,
                    )
                    continue

                # Subset to this domain AND this sample
                combined_mask = (
                    sample_mask
                    & (
                        self.adata.obs[
                            self.clustering_col
                        ]
                        == domain
                    )
                )
                d_adata = self.adata[
                    combined_mask
                ].copy()

                if d_adata.n_obs < 10:
                    ax.text(
                        0.5,
                        0.5,
                        f"Too few cells\n(n={d_adata.n_obs})",
                        transform=ax.transAxes,
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="gray",
                    )
                    ax.set_title(
                        f"Domain {domain}",
                        fontsize=8,
                    )
                    continue

                # Build spatial neighbors within
                # this sample only
                effective_neighs = min(
                    n_neighs,
                    d_adata.n_obs - 1,
                )
                sq.gr.spatial_neighbors(
                    d_adata,
                    coord_type="generic",
                    n_neighs=effective_neighs,
                )

                # Get score columns
                score_cols = self._get_score_cols_for_domain(
                    d_adata,
                    sig_celltypes,
                    method,
                )

                if not score_cols:
                    ax.text(
                        0.5,
                        0.5,
                        "No score columns\navailable",
                        transform=ax.transAxes,
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="gray",
                    )
                    ax.set_title(
                        f"Domain {domain}",
                        fontsize=8,
                    )
                    continue

                # Compute Moran's I
                sq.gr.spatial_autocorr(
                    d_adata,
                    genes=score_cols,
                    mode="moran",
                    attr="obs",
                    n_perms=n_perms,
                )

                morans = d_adata.uns[
                    "moranI"
                ].loc[score_cols]
                morans = (
                    morans.sort_values(
                        "I",
                        ascending=False,
                    )
                )

                # Clean labels
                clean_labels = morans.index.str.replace(
                    "_score$",
                    "",
                    regex=True,
                )

                # Colour: selected vs not
                colors = [
                    "steelblue"
                    if lbl in selected
                    else "silver"
                    for lbl in clean_labels
                ]
                edge_colors = [
                    "black"
                    if lbl in selected
                    else "gray"
                    for lbl in clean_labels
                ]

                x_pos = np.arange(
                    len(morans)
                )
                ax.bar(
                    x_pos,
                    morans["I"].values,
                    color=colors,
                    edgecolor=edge_colors,
                    linewidth=0.5,
                    zorder=3,
                )

                # Significance markers
                for i, (
                    idx,
                    row,
                ) in enumerate(
                    morans.iterrows()
                ):
                    pval = row.get(
                        "pval_norm",
                        row.get(
                            "pval", 1.0
                        ),
                    )
                    if pval < 0.001:
                        sig_text = "***"
                    elif pval < 0.01:
                        sig_text = "**"
                    elif pval < 0.05:
                        sig_text = "*"
                    else:
                        sig_text = ""
                    if sig_text:
                        y_pos = max(
                            row["I"]
                            + 0.01,
                            0.02,
                        )
                        ax.text(
                            i,
                            y_pos,
                            sig_text,
                            ha="center",
                            va="bottom",
                            fontsize=5,
                            color="black",
                        )

                ax.set_xticks(x_pos)
                ax.set_xticklabels(
                    clean_labels,
                    rotation=60,
                    ha="right",
                    fontsize=5,
                )
                ax.set_ylabel(
                    "Moran's I",
                    fontsize=7,
                )
                ax.set_title(
                    f"Domain {domain}",
                    fontsize=8,
                )
                ax.tick_params(
                    labelsize=6
                )
                ax.axhline(
                    0,
                    color="black",
                    linewidth=0.5,
                    zorder=1,
                )
                ax.grid(
                    axis="y",
                    alpha=0.3,
                    linestyle="--",
                    zorder=0,
                )

            # Turn off unused axes
            for ax_idx in range(
                n_domains,
                len(axes_flat),
            ):
                axes_flat[ax_idx].axis(
                    "off"
                )

            fig.suptitle(
                f"Moran's I — {method} — {self.clustering_col} — {sample_id}",
                fontsize=11,
            )
            plt.tight_layout()

            if save:
                path = (
                    self.output_dir
                    / "figures"
                    / f"gmm_morans_{method}_{self.clustering_col}_{sample_id}.png"
                )
                plt.savefig(
                    path,
                    dpi=400,
                    bbox_inches="tight",
                )
                plt.close()
                logger.info(
                    f"Saved Moran's I plot to {path}"
                )
            else:
                plt.show()

    def plot_gmm_proportions(
        self,
        method: str = "enrichmap",
        save: bool = True,
    ) -> None:
        """
        Per-cell argmax proportion bar plot per domain,
        using only cell type scores whose rankings have
        meanchange > 0 and padj < 0.05 (same filter as
        the elbow plot).

        For each domain, every cell is assigned to its
        highest-scoring (significant) cell type, and the
        fraction of cells per type is plotted as a bar chart.

        Produces a grid of subplots (one per domain) matching
        the layout of the other diagnostic plots.

        Must be called after assign_cell_types() with
        annotation_method='gmm'.
        """
        if method == "enrichmap":
            debug_info = self._gmm_debug_info_enrichmap
        else:
            raise ValueError(
                f"Unknown method '{method}'"
            )

        if not debug_info:
            raise ValueError(
                f"No GMM debug info for '{method}'. "
                "Run assign_cell_types() with "
                "annotation_method='gmm' first."
            )

        # Plot domains that have significant cell types
        # (domain_data is already filtered to
        # meanchange > 0 and padj < 0.05)
        plottable = {
            d: info
            for d, info in debug_info.items()
            if info.get("domain_data")
            is not None
            and len(
                info.get(
                    "domain_data", []
                )
            )
            > 0
        }

        if not plottable:
            logger.warning(
                "No domains with significant data to plot proportions."
            )
            return

        domains = list(plottable.keys())
        n_domains = len(domains)
        n_cols = min(4, n_domains)
        n_rows = int(
            np.ceil(n_domains / n_cols)
        )

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(
                5 * n_cols,
                4 * n_rows,
            ),
        )
        axes_flat = np.array(
            axes
        ).flatten()

        for ax_idx, domain in enumerate(
            domains
        ):
            ax = axes_flat[ax_idx]
            info = plottable[domain]
            domain_data = info[
                "domain_data"
            ]
            selected = set(
                info["selected"]
            )

            # Get cell type names that pass the filter
            sig_celltypes = domain_data[
                "name"
            ].values

            if len(sig_celltypes) == 0:
                ax.text(
                    0.5,
                    0.5,
                    "No significant\ncell types",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="gray",
                )
                ax.set_title(
                    f"Domain {domain}",
                    fontsize=8,
                )
                continue

            # Subset adata to cells in this domain
            domain_mask = (
                self.adata.obs[
                    self.clustering_col
                ]
                == domain
            )

            # Get scores for sig cell types only
            if method == "enrichmap":
                score_cols = [
                    f"{ct}_score"
                    for ct in sig_celltypes
                    if f"{ct}_score"
                    in self.adata.obs.columns
                ]
                if not score_cols:
                    if (
                        self.enrichmap_scores
                        is not None
                    ):
                        avail = self.enrichmap_scores.columns
                        score_cols = [
                            c
                            for c in avail
                            if c.replace(
                                "_score",
                                "",
                            )
                            in sig_celltypes
                        ]
                    else:
                        score_cols = []

                if not score_cols:
                    ax.text(
                        0.5,
                        0.5,
                        "No score columns\navailable",
                        transform=ax.transAxes,
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="gray",
                    )
                    ax.set_title(
                        f"Domain {domain}",
                        fontsize=8,
                    )
                    continue

                # Get scores for domain cells
                if (
                    score_cols[0]
                    in self.adata.obs.columns
                ):
                    domain_scores = self.adata.obs.loc[
                        domain_mask,
                        score_cols,
                    ]
                elif (
                    self.enrichmap_scores
                    is not None
                ):
                    domain_scores = self.enrichmap_scores.loc[
                        domain_mask,
                        score_cols,
                    ]
                else:
                    continue

                # Argmax per cell
                top_per_cell = domain_scores.idxmax(
                    axis=1
                )
                # Clean label: remove _score
                top_per_cell = top_per_cell.str.replace(
                    "_score$",
                    "",
                    regex=True,
                )

            # Compute proportions
            props = top_per_cell.value_counts(
                normalize=True
            ).sort_values(
                ascending=False
            )

            x_pos = np.arange(
                len(props)
            )

            # Colour: selected (steelblue) vs not (silver)
            colors = [
                "steelblue"
                if lbl in selected
                else "silver"
                for lbl in props.index
            ]
            edge_colors = [
                "black"
                if lbl in selected
                else "gray"
                for lbl in props.index
            ]

            ax.bar(
                x_pos,
                props.values,
                color=colors,
                edgecolor=edge_colors,
                linewidth=0.5,
                zorder=3,
            )

            ax.set_xticks(x_pos)
            ax.set_xticklabels(
                props.index,
                rotation=60,
                ha="right",
                fontsize=5,
            )
            ax.set_ylabel(
                "Fraction of cells",
                fontsize=7,
            )
            ax.set_title(
                f"Domain {domain}",
                fontsize=8,
            )
            ax.tick_params(labelsize=6)
            ax.grid(
                axis="y",
                alpha=0.3,
                linestyle="--",
                zorder=0,
            )

        # Turn off unused axes
        for ax_idx in range(
            n_domains, len(axes_flat)
        ):
            axes_flat[ax_idx].axis(
                "off"
            )

        fig.suptitle(
            f"Cell type proportions (argmax) — {method} — {self.clustering_col}",
            fontsize=11,
        )
        plt.tight_layout()

        if save:
            path = (
                self.output_dir
                / "figures"
                / f"gmm_proportions_{method}_{self.clustering_col}.png"
            )
            plt.savefig(
                path,
                dpi=400,
                bbox_inches="tight",
            )
            plt.close()
            logger.info(
                f"Saved proportions plot to {path}"
            )
        else:
            plt.show()

    def plot_gmm_correlation(
        self,
        method: str = "enrichmap",
        save: bool = True,
    ) -> None:
        """
        Per-domain Spearman correlation matrix (lower triangle)
        of cell type scores across cells, using only cell types
        with meanchange > 0 and padj < 0.05.

        Low/negative correlation between two cell types means
        they are driven by different cells (real heterogeneity).
        High positive correlation means the same cells score
        high for both (marker overlap).
        """
        if method == "enrichmap":
            debug_info = self._gmm_debug_info_enrichmap
        else:
            raise ValueError(
                f"Unknown method '{method}'"
            )

        if not debug_info:
            raise ValueError(
                f"No GMM debug info for '{method}'. "
                "Run assign_cell_types() with "
                "annotation_method='gmm' first."
            )

        plottable = {
            d: info
            for d, info in debug_info.items()
            if info.get("domain_data")
            is not None
            and len(
                info.get(
                    "domain_data", []
                )
            )
            > 1  # Need at least 2 cell types for correlation
        }

        if not plottable:
            logger.warning(
                "No domains with >= 2 significant cell types to plot correlation."
            )
            return

        domains = list(plottable.keys())
        n_domains = len(domains)
        n_cols = min(4, n_domains)
        n_rows = int(
            np.ceil(n_domains / n_cols)
        )

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(
                5 * n_cols,
                4.5 * n_rows,
            ),
        )
        axes_flat = np.array(
            axes
        ).flatten()

        for ax_idx, domain in enumerate(
            domains
        ):
            ax = axes_flat[ax_idx]
            info = plottable[domain]
            domain_data = info[
                "domain_data"
            ]
            sig_celltypes = domain_data[
                "name"
            ].values

            # Subset adata to cells in this domain
            domain_mask = (
                self.adata.obs[
                    self.clustering_col
                ]
                == domain
            )

            # Get scores
            if method == "enrichmap":
                score_cols = [
                    f"{ct}_score"
                    for ct in sig_celltypes
                    if f"{ct}_score"
                    in self.adata.obs.columns
                ]
                if not score_cols:
                    if (
                        self.enrichmap_scores
                        is not None
                    ):
                        avail = self.enrichmap_scores.columns
                        score_cols = [
                            c
                            for c in avail
                            if c.replace(
                                "_score",
                                "",
                            )
                            in sig_celltypes
                        ]
                    else:
                        score_cols = []

                if len(score_cols) < 2:
                    ax.text(
                        0.5,
                        0.5,
                        "< 2 score columns",
                        transform=ax.transAxes,
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="gray",
                    )
                    ax.set_title(
                        f"Domain {domain}",
                        fontsize=8,
                    )
                    continue

                if (
                    score_cols[0]
                    in self.adata.obs.columns
                ):
                    domain_scores = self.adata.obs.loc[
                        domain_mask,
                        score_cols,
                    ]
                elif (
                    self.enrichmap_scores
                    is not None
                ):
                    domain_scores = self.enrichmap_scores.loc[
                        domain_mask,
                        score_cols,
                    ]
                else:
                    continue

                # Clean column names
                domain_scores = domain_scores.rename(
                    columns=lambda c: (
                        c.replace(
                            "_score", ""
                        )
                    )
                )

            # Compute Spearman correlation
            corr = domain_scores.corr(
                method="spearman"
            )

            # Lower triangle mask
            mask = np.triu(
                np.ones_like(
                    corr, dtype=bool
                ),
                k=1,
            )

            n_types = len(corr)
            # Adaptive font size for annotations
            if n_types <= 5:
                annot_fontsize = 6
            elif n_types <= 10:
                annot_fontsize = 4.5
            else:
                annot_fontsize = 3.5

            sns.heatmap(
                corr,
                mask=mask,
                annot=True,
                fmt=".2f",
                cmap="RdBu_r",
                center=0,
                vmin=-1,
                vmax=1,
                square=True,
                ax=ax,
                cbar_kws={
                    "shrink": 0.6,
                    "aspect": 15,
                },
                annot_kws={
                    "fontsize": annot_fontsize,
                },
                linewidths=0.5,
                linecolor="white",
            )

            ax.set_xticklabels(
                ax.get_xticklabels(),
                rotation=60,
                ha="right",
                fontsize=5,
            )
            ax.set_yticklabels(
                ax.get_yticklabels(),
                rotation=0,
                fontsize=5,
            )
            ax.set_title(
                f"Domain {domain}",
                fontsize=8,
            )

        # Turn off unused axes
        for ax_idx in range(
            n_domains, len(axes_flat)
        ):
            axes_flat[ax_idx].axis(
                "off"
            )

        fig.suptitle(
            f"Score correlation (Spearman) — {method} — {self.clustering_col}",
            fontsize=11,
        )
        plt.tight_layout()

        if save:
            path = (
                self.output_dir
                / "figures"
                / f"gmm_correlation_{method}_{self.clustering_col}.png"
            )
            plt.savefig(
                path,
                dpi=400,
                bbox_inches="tight",
            )
            plt.close()
            logger.info(
                f"Saved correlation plot to {path}"
            )
        else:
            plt.show()

    def plot_gmm_entropy(
        self,
        method: str = "enrichmap",
        save: bool = True,
    ) -> None:
        """
        Single bar plot showing Shannon entropy of the
        argmax cell type proportions for each domain.

        High entropy = highly heterogeneous (many cell types
        at similar fractions). Low entropy = dominated by
        one cell type.

        Uses only cell types with meanchange > 0 and
        padj < 0.05 for the argmax computation.
        """
        if method == "enrichmap":
            debug_info = self._gmm_debug_info_enrichmap
        else:
            raise ValueError(
                f"Unknown method '{method}'"
            )

        if not debug_info:
            raise ValueError(
                f"No GMM debug info for '{method}'. "
                "Run assign_cell_types() with "
                "annotation_method='gmm' first."
            )

        plottable = {
            d: info
            for d, info in debug_info.items()
            if info.get("domain_data")
            is not None
            and len(
                info.get(
                    "domain_data", []
                )
            )
            > 0
        }

        if not plottable:
            logger.warning(
                "No domains with significant data to plot entropy."
            )
            return

        entropy_data = {}

        for (
            domain,
            info,
        ) in plottable.items():
            domain_data = info[
                "domain_data"
            ]
            selected = set(
                info["selected"]
            )
            sig_celltypes = domain_data[
                "name"
            ].values

            if len(sig_celltypes) == 0:
                continue

            domain_mask = (
                self.adata.obs[
                    self.clustering_col
                ]
                == domain
            )

            # Get scores for sig cell types
            if method == "enrichmap":
                score_cols = [
                    f"{ct}_score"
                    for ct in sig_celltypes
                    if f"{ct}_score"
                    in self.adata.obs.columns
                ]
                if not score_cols:
                    if (
                        self.enrichmap_scores
                        is not None
                    ):
                        avail = self.enrichmap_scores.columns
                        score_cols = [
                            c
                            for c in avail
                            if c.replace(
                                "_score",
                                "",
                            )
                            in sig_celltypes
                        ]
                    else:
                        score_cols = []

                if not score_cols:
                    continue

                if (
                    score_cols[0]
                    in self.adata.obs.columns
                ):
                    domain_scores = self.adata.obs.loc[
                        domain_mask,
                        score_cols,
                    ]
                elif (
                    self.enrichmap_scores
                    is not None
                ):
                    domain_scores = self.enrichmap_scores.loc[
                        domain_mask,
                        score_cols,
                    ]
                else:
                    continue

                top_per_cell = domain_scores.idxmax(
                    axis=1
                ).str.replace(
                    "_score$",
                    "",
                    regex=True,
                )

            # Compute Shannon entropy
            props = top_per_cell.value_counts(
                normalize=True
            )
            # H = -sum(p * log2(p))
            H = -(
                props * np.log2(props)
            ).sum()
            # Normalize by max possible entropy
            n_types = len(props)
            H_max = (
                np.log2(n_types)
                if n_types > 1
                else 1.0
            )
            H_norm = H / H_max

            entropy_data[domain] = {
                "H": H,
                "H_norm": H_norm,
                "n_types": n_types,
                "n_selected": len(
                    selected
                ),
            }

        if not entropy_data:
            logger.warning(
                "No entropy data to plot."
            )
            return

        # Sort by normalized entropy descending
        sorted_domains = sorted(
            entropy_data.keys(),
            key=lambda d: entropy_data[
                d
            ]["H_norm"],
            reverse=True,
        )

        h_norm_vals = [
            entropy_data[d]["H_norm"]
            for d in sorted_domains
        ]
        n_selected_vals = [
            entropy_data[d][
                "n_selected"
            ]
            for d in sorted_domains
        ]

        # Colour by number of selected cell types
        # More selected types → darker blue
        max_sel = max(n_selected_vals)
        colors = [
            plt.cm.Blues(
                0.3
                + 0.7
                * (
                    n / max_sel
                    if max_sel > 0
                    else 0
                )
            )
            for n in n_selected_vals
        ]

        fig, ax = plt.subplots(
            figsize=(
                max(
                    8,
                    len(sorted_domains)
                    * 0.5,
                ),
                4,
            )
        )

        x_pos = np.arange(
            len(sorted_domains)
        )
        ax.bar(
            x_pos,
            h_norm_vals,
            color=colors,
            edgecolor="black",
            linewidth=0.5,
            zorder=3,
        )

        # Annotate number of selected cell types
        for i, (h, n_sel) in enumerate(
            zip(
                h_norm_vals,
                n_selected_vals,
            )
        ):
            ax.text(
                i,
                h + 0.01,
                f"n={n_sel}",
                ha="center",
                va="bottom",
                fontsize=5,
                color="gray",
            )

        ax.set_xticks(x_pos)
        ax.set_xticklabels(
            sorted_domains,
            rotation=60,
            ha="right",
            fontsize=6,
        )
        ax.set_ylabel(
            "Normalized entropy (H / H_max)",
            fontsize=8,
        )
        ax.set_ylim(0, 1.15)
        ax.axhline(
            1.0,
            color="red",
            ls="--",
            lw=0.8,
            alpha=0.5,
            label="Max entropy (uniform)",
        )
        ax.axhline(
            0.5,
            color="gray",
            ls=":",
            lw=0.8,
            alpha=0.5,
        )
        ax.tick_params(labelsize=6)
        ax.grid(
            axis="y",
            alpha=0.3,
            linestyle="--",
            zorder=0,
        )
        ax.legend(fontsize=6)

        fig.suptitle(
            f"Domain heterogeneity (entropy) — {method} — {self.clustering_col}",
            fontsize=11,
        )
        plt.tight_layout()

        if save:
            path = (
                self.output_dir
                / "figures"
                / f"gmm_entropy_{method}_{self.clustering_col}.png"
            )
            plt.savefig(
                path,
                dpi=400,
                bbox_inches="tight",
            )
            plt.close()
            logger.info(
                f"Saved entropy plot to {path}"
            )
        else:
            plt.show()

    def run_full_pipeline(
        self,
        n_top: int | str = 6,
        save_plots: bool = True,
        save_final: bool = True,
    ) -> AnnData:
        """
        Run the full annotation pipeline.

        1. Run Enrichmap scoring
        2. Assign cell types
        3. Save plots
        4. Optionally save annotated adata to disk (save_final=True)
        """
        self.run_enrichmap(save=False)
        self._save_enrichmap_plots()
        self.assign_cell_types(
            n_top=n_top
        )
        if save_plots:
            self.plot_matrixplot(
                method="enrichmap",
                n_top=5,
            )
            self.plot_spatial(
                color_col="cell_type_enrichmap",
                spot_size=9,
            )
            self.plot_spatial(
                color_col="cell_type_enrichmap_high_granularity",
                spot_size=9,
            )
            if (
                self.annotation_method
                == "gmm"
            ):
                if self._gmm_debug_info_enrichmap:
                    self.plot_gmm_diagnostics(
                        method="enrichmap"
                    )
                    self.plot_gmm_elbow(
                        method="enrichmap"
                    )
                    self.plot_gmm_morans(
                        method="enrichmap"
                    )
                    self.plot_gmm_proportions(
                        method="enrichmap"
                    )
                    self.plot_gmm_correlation(
                        method="enrichmap"
                    )
                    self.plot_gmm_entropy(
                        method="enrichmap"
                    )

        if save_final:
            adata_to_save = (
                prepare_for_saving(
                    self.adata.copy()
                )
            )
            adata_to_save.write_h5ad(
                self.output_dir
                / f"adata_annotated_{self.clustering_col}.h5ad"
            )
            logger.info(
                f"Saved annotated adata to {self.output_dir / f'adata_annotated_{self.clustering_col}.h5ad'}"
            )

        return self.adata


# %%

cfg = load_config(
    r"/mnt/work/RO_src/Projects/Paper_ST_and_scRNA/spatial/scripts/config.yaml"
)
sample_folders = cfg["sample_folders"]
id_to_image = cfg["id_to_image"]
sample_paths = cfg["sample_paths"]
sample_mapping = cfg["sample_mapping"]
marker_genes_dict = cfg[
    "marker_genes"
]  # marker_genes_grouped, marker_genes

FIGSIZE = (3, 3)
rcParams["figure.figsize"] = FIGSIZE
mpl.rcParams["figure.dpi"] = 600

plt.style.use("bmh")
plt.rcParams.update(
    {
        "figure.figsize": (12, 8),
        "axes.facecolor": "white",
        "axes.edgecolor": "black",
    }
)
# Can also check OTMODE github to see an example of processing steps for scrna

# Open json file and select some of the pathways
hallmarks = json.load(
    open(
        "/mnt/work/RO_src/Projects/Paper_ST_and_scRNA/spatial/data/files/h.all.v2025.1.Hs.json"
    )
)
filtered = [
    # "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION",
    "HALLMARK_TNFA_SIGNALING_VIA_NFKB",
    "HALLMARK_IL6_JAK_STAT3_SIGNALING",
    "HALLMARK_NOTCH_SIGNALING",
    "HALLMARK_INTERFERON_ALPHA_RESPONSE",
    "HALLMARK_INTERFERON_GAMMA_RESPONSE",
    "HALLMARK_COMPLEMENT",
    "HALLMARK_INFLAMMATORY_RESPONSE",
    "HALLMARK_IL2_STAT5_SIGNALING",
]

# SUbset hallmarks keys to filtered list
filtered_hallmarks = {}
for (
    hallmark,
    genes,
) in hallmarks.items():
    if hallmark in filtered:
        filtered_hallmarks[hallmark] = [
            g
            for g in genes[
                "geneSymbols"
            ]
        ]

# Create output directory for figures
figures_dir = "figures_novae"
os.makedirs(figures_dir, exist_ok=True)

run_dir = "novae_05022026_1717"


# %%
# def plot_jaccard_index(dict):

#     import itertools
#     import pandas as pd
#     import seaborn as sns
#     import matplotlib.pyplot as plt

#     from scipy.spatial.distance import squareform, pdist

#     all_genes = sorted(set().union(*marker_genes_dict.values()))
#     cell_types = list(marker_genes_dict.keys())
#     binary = np.array(
#         [[g in marker_genes_dict[ct] for g in all_genes] for ct in cell_types]
#     )

#     jaccard_matrix = pd.DataFrame(
#         1 - squareform(pdist(binary, metric="jaccard")),
#         index=cell_types,
#         columns=cell_types,
#     )

#     # Plot
#     fig, ax = plt.subplots(figsize=(10, 8))
#     sns.heatmap(
#         jaccard_matrix.astype(float),
#         annot=True,
#         fmt=".2f",
#         cmap="YlOrRd",
#         vmin=0,
#         vmax=1,
#         square=True,
#         linewidths=0.5,
#         ax=ax,
#     )
#     ax.set_title("Pairwise Jaccard Similarity of Marker Gene Sets")
#     plt.tight_layout()
#     plt.show()

#     # Show overlapping genes for pairs above threshold
#     threshold = 0.1
#     for ct_a, ct_b in itertools.combinations(cell_types, 2):
#         overlap = set(marker_genes_dict[ct_a]) & set(marker_genes_dict[ct_b])
#         jac = jaccard_matrix.loc[ct_a, ct_b]
#         if jac > threshold:
#             print(f"{ct_a} vs {ct_b} — Jaccard={jac:.2f}, shared: {overlap}")


# plot_jaccard_index(marker_genes_dict)

# %%
adatas = []
for sample in sample_folders:
    data = ad.read_h5ad(
        f"/mnt/archive/RO_src/data/spatial/processed/{run_dir}/{sample}.h5ad"
    )
    data = revert_from_conversion(data)

    adatas.append(data)

adata_concat = ad.concat(
    adatas,
    label="sample_id",
    keys=sample_folders,
    join="inner",
    merge="same",
)


# %%

# data = ad.read_h5ad(
#     f"/mnt/archive/RO_src/data/spatial/processed/{run_dir}/adata_annotated_all_domains.h5ad"
# )
# data = revert_from_conversion(data)

# n_sample = 10000
# sampled_idx = (
#     adata_concat.obs.groupby(
#         "novae_domains_26",
#         group_keys=False,
#     )
#     .apply(
#         lambda x: x.sample(
#             frac=n_sample / adata_concat.n_obs,
#             random_state=42,
#         )
#     )
#     .index
# )
# adata_concat = adata_concat[sampled_idx].copy()

# %%
# https://www.one-tab.com/page/OAyZkctSQimPpKR7dd76Eg
# Initialize the annotator
# domains_input accepts either a single int (e.g. 5) or an iterable (e.g. range(5, 40))
domains_input = range(5, 41)
domain_nums = (
    [domains_input]
    if isinstance(domains_input, int)
    else domains_input
)

logger.info(
    f"Will be running CellAnnotator for domains {domain_nums}"
)
output_dir = f"/mnt/archive/RO_src/data/spatial/processed/{run_dir}"

for n in domain_nums:
    domain = f"novae_domains_{n}"

    logger.info(
        f"Running CellAnnotator for {domain}"
    )
    annotator = CellTypeAnnotator(
        adata=adata_concat,
        marker_genes=marker_genes_dict,
        domains_prop_filter=50,  # original was 0.001
        clustering_col=domain,
        output_dir=output_dir,
        batch_key="sample_id",
        annotation_method="gmm",
        one_vs_all_test="wilcoxon",
    )
    annotator.run_full_pipeline(
        n_top="all",
        save_plots=True,
        save_final=False,
    )

    # Copy annotation columns back to adata_concat with domain-specific names
    col = "cell_type_enrichmap"
    if (
        col
        in annotator.adata.obs.columns
    ):
        adata_concat.obs[
            f"{col}_{domain}"
        ] = annotator.adata.obs[col]

    col_hg = "cell_type_enrichmap_high_granularity"
    if (
        col_hg
        in annotator.adata.obs.columns
    ):
        adata_concat.obs[
            f"{col_hg}_{domain}"
        ] = annotator.adata.obs[col_hg]

    # Copy scoring matrices with domain-specific keys into obsm
    # Reindex to adata_concat.obs_names (annotator may have fewer cells
    # due to filtering low-proportion domains)
    if (
        annotator.enrichmap_scores
        is not None
    ):
        adata_concat.obsm[
            f"score_enrichmap_{domain}"
        ] = annotator.enrichmap_scores.reindex(
            adata_concat.obs_names,
            fill_value=0,
        )

# %%
# Save a single h5ad with all domain annotation columns
adata_to_save = prepare_for_saving(
    adata_concat.copy()
)
adata_to_save.write_h5ad(
    Path(output_dir)
    / "adata_annotated_all_domains.h5ad"
)
logger.info(
    f"Saved combined annotated adata to {output_dir}/adata_annotated_all_domains.h5ad"
)

# # %%
# data = ad.read_h5ad(
#     f"/mnt/archive/RO_src/data/spatial/processed/{run_dir}/adata_annotated_all_domains.h5ad"
# )
# data = revert_from_conversion(data)


# # %%
# for sample in data.obs[
#     "sample_id"
# ].unique():
#     adata_sample = data[
#         data.obs["sample_id"] == sample
#     ].copy()

#     sc.pl.spatial(
#         adata_sample,
#         color=[
#             "cell_type_enrichmap_high_granularity_novae_domains_26",
#         ],
#         cmap="RdBu_r",
#         spot_size=9,
#     )
#     em.pl.gene_contributions_pca(
#         adata_sample,
#         score_key="Smooth_Muscle_score",
#         top_n_genes=3,
#     )

#     em.pl.morans_correlogram(
#         adata_sample,
#         score_key="Smooth_Muscle_score",
#     )

#     em.pl.variogram(
#         adata_sample,
#         score_keys=[
#             "Smooth_Muscle_score",
#             "Epithelial_score",
#         ],
#     )


# # %%
# em.pl.signature_correlation_heatmap(
#     data,
#     score_keys=[
#         "Smooth_Muscle_score",
#         "Epithelial_score",
#         "Fibroblast_score",
#     ],
#     library_key="sample_id",
# )

# # %%
# em.pl.cross_moran_scatter(
#     data,
#     score_x="Smooth_Muscle_score",
#     score_y="Epithelial_score",
#     library_key="sample_id",
# )

# # %%

# for sample in adata_concat.obs["sample_id"].unique():
#     adata_sample = adata_concat[adata_concat.obs["sample_id"] == sample].copy()

#     novae.plot.spatially_variable_genes(
#         adata_sample,
#         obs_key="novae_domains_20",
#         top_k=10,
#         vmax="p95",
#         cell_size=8,
#         show=False,
#     )
#     plt.savefig(
#         f"spatially_variable_genes_{sample}.png",
#         dpi=900,
#         bbox_inches="tight",
#     )
#     plt.close()

# # %%
# sc.tl.rank_genes_groups(
#     adata=adata_concat,
#     groupby=clustering_col,
#     method="wilcoxon",
# )
# sc.pl.rank_genes_groups_dotplot(
#     adata=adata_concat,
#     groupby=clustering_col,
#     standard_scale="var",
#     n_genes=5,
# )
# df_marker_genes = (
#     sc.get.rank_genes_groups_df(
#         adata=adata_concat,
#         group=None,
#         pval_cutoff=0.05,
#     )
# )
# df_marker_genes.to_csv(
#     "/mnt/archive/RO_src/data/spatial/processed/novae_22012026_1329/marker_genes.csv"
# )

# %%
