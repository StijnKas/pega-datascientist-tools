from __future__ import annotations

import datetime
import logging
import os
from pathlib import Path
from typing import Callable, Iterable, Optional, Tuple, Union

import polars as pl

from .. import pega_io
from ..pega_io.File import read_ds_export
from ..utils import cdh_utils
from ..utils.cdh_utils import _polars_capitalize
from ..utils.types import QUERY
from .ADMTrees import AGB
from .Aggregates import Aggregates
from .BinAggregator import BinAggregator
from .Plots import Plots
from .Reports import Reports

logger = logging.getLogger(__name__)


class ADMDatamart:
    model_data: Optional[pl.LazyFrame]
    predictor_data: Optional[pl.LazyFrame]
    combined_data: Optional[pl.LazyFrame]

    def __init__(
        self,
        model_df: Optional[pl.LazyFrame],
        predictor_df: Optional[pl.LazyFrame],
        *,
        query: Optional[QUERY] = None,
        extract_pyname_keys: bool = True,
    ) -> None:
        self.context_keys: list = [
            "Channel",
            "Direction",
            "Issue",
            "Group",
            "Name",
        ]

        self.plot = Plots(datamart=self)
        self.aggregates = Aggregates(datamart=self)
        self.agb = AGB(datamart=self)
        self.generate = Reports(datamart=self)
        self.bin_aggregator = BinAggregator(dm=self)

        self.model_data = self._validate_model_data(
            model_df, query=query, extract_pyname_keys=extract_pyname_keys
        )
        self.predictor_data = self._validate_predictor_data(predictor_df)

        self.combined_data = self.aggregates._combine_data(
            self.model_data, self.predictor_data
        )

    @classmethod
    def from_ds_export(
        cls,
        model_filename: Optional[str] = None,
        predictor_filename: Optional[str] = None,
        base_path: str = ".",
        *,
        query: Optional[Union[pl.Expr, Iterable[pl.Expr]]] = None,
        extract_pyname_keys: bool = True,
    ):
        model_df = read_ds_export(model_filename or "model_data", base_path)
        predictor_df = read_ds_export(predictor_filename or "predictor_data", base_path)
        return cls(
            model_df, predictor_df, query=query, extract_pyname_keys=extract_pyname_keys
        )

    @classmethod
    def from_s3(cls): ...

    def _validate_model_data(
        self,
        df: Optional[pl.LazyFrame],
        query: Optional[QUERY] = None,
        extract_pyname_keys: bool = True,
    ) -> Optional[pl.LazyFrame]:
        if df is None:
            logger.info("No model data available.")
            return df

        df = _polars_capitalize(df)
        schema = df.collect_schema()
        if extract_pyname_keys and "Name" in schema.names():
            df = cdh_utils._extract_keys(df)

        if "Treatment" in schema.names():
            self.context_keys.append("Treatment")

        self.context_keys = [k for k in self.context_keys if k in schema.names()]

        df = df.with_columns(
            SuccessRate=(pl.col("Positives") / pl.col("ResponseCount")).fill_nan(
                pl.lit(0)
            ),
        )
        if not isinstance(schema["SnapshotTime"], pl.Datetime):
            df = df.with_columns(SnapshotTime=cdh_utils.parse_pega_date_time_formats())

        return cdh_utils._apply_query(df, query)

    def _validate_predictor_data(
        self, df: Optional[pl.LazyFrame]
    ) -> Optional[pl.LazyFrame]:
        if df is None:
            logger.info("No predictor data available.")
            return df
        df = _polars_capitalize(df)
        schema = df.collect_schema()

        if "BinResponseCount" not in schema.names():  # pragma: no cover
            df = df.with_columns(
                BinResponseCount=(pl.col("BinPositives") + pl.col("BinNegatives"))
            )
        df = df.with_columns(
            BinPropensity=pl.col("BinPositives") / pl.col("BinResponseCount"),
            BinAdjustedPropensity=(
                (pl.col("BinPositives") + pl.lit(0.5))
                / (pl.col("BinResponseCount") + pl.lit(1))
            ),
        )

        if "PredictorCategory" not in schema.names():
            df = self.apply_predictor_categorization(df)

        return df

    def apply_predictor_categorization(
        self,
        df: pl.LazyFrame,
        categorization: Optional[
            Union[pl.Expr, Callable[..., pl.Expr]]
        ] = cdh_utils.default_predictor_categorization,
    ) -> pl.LazyFrame:
        if callable(categorization):
            categorization = categorization()

        return df.with_columns(PredictorCategory=categorization)

    def save_data(
        self, path: os.StrPath = "."
    ) -> Tuple[Optional[Path], Optional[Path]]:
        """Cache modelData and predictorData to files.

        Parameters
        ----------
        path : str
            Where to place the files

        Returns
        -------
        (Optional[Path], Optional[Path]):
            The paths to the model and predictor data files
        """
        time = datetime.datetime.now().strftime("%Y%m%dT%H%M%S.%f")[:-3]
        modeldata_cache, predictordata_cache = None, None
        if self.model_data is not None:
            modeldata_cache = pega_io.cache_to_file(
                self.model_data, path, name=f"cached_model_data_{time}"
            )
        if self.predictor_data is not None:
            predictordata_cache = pega_io.cache_to_file(
                self.predictor_data, path, name=f"cached_predictor_data_{time}"
            )

        return modeldata_cache, predictordata_cache