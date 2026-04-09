"""Shared helpers for validation curve plotting and CSV export scripts."""

from __future__ import annotations

import pathlib

import pandas as pd


def load_validation_csv(snapshot: pathlib.Path, metric: str) -> pd.DataFrame:
    csv_path = snapshot / "validation_results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing validation_results.csv in {snapshot}")

    df = pd.read_csv(csv_path)
    if "validation_scope" not in df.columns or metric not in df.columns:
        raise ValueError(f"Unexpected CSV format in {csv_path}")

    df["training_steps"] = pd.to_numeric(df["training_steps"], errors="coerce")
    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df["validation_scope"] = df["validation_scope"].astype(str)
    return df


def apply_scope(series: pd.DataFrame, scope: str, x_axis: str) -> pd.DataFrame:
    if series.empty:
        return series
    if scope == "all":
        return series
    if scope in {"step", "epoch", "initial"}:
        return series[series["validation_scope"] == scope].copy()

    # auto: keep one row per x-axis, preferring denser "step" logs, then epoch, then initial.
    priority = {"step": 0, "epoch": 1, "initial": 2}
    scoped = series.copy()
    scoped["_scope_pri"] = scoped["validation_scope"].map(priority).fillna(99).astype(int)
    scoped = scoped.sort_values([x_axis, "_scope_pri"])
    scoped = scoped.groupby(x_axis, as_index=False).first()
    if "_scope_pri" in scoped.columns:
        scoped = scoped.drop(columns=["_scope_pri"])
    return scoped


def add_step_zero(df: pd.DataFrame, x_axis: str, metric: str) -> pd.DataFrame:
    if x_axis != "training_steps" or (df[x_axis] == 0).any():
        return df
    if df.empty:
        return df

    first = df.sort_values(x_axis).iloc[[0]].copy()
    first[x_axis] = 0
    if pd.isna(first[metric].iloc[0]):
        return df
    return pd.concat([first, df], ignore_index=True)


def prepare_series(
    df: pd.DataFrame,
    benchmark: str,
    args,
    *,
    include_zero: bool = False,
    include_explicit_zero: bool = True,
) -> pd.DataFrame:
    series = df[df["benchmark"] == benchmark].copy()
    if args.target is not None:
        series = series[series["validation_target"] == args.target]
    series = series.sort_values(args.x_axis).dropna(subset=[args.x_axis, args.metric])
    series = apply_scope(series, args.scope, args.x_axis)
    if args.exclude_step_zero:
        series = series[series[args.x_axis] != 0]
    if not include_explicit_zero:
        series = series[series[args.x_axis] != 0]
    if include_zero:
        series = add_step_zero(series, args.x_axis, args.metric)
    return series
