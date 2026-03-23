from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd


def select_random(df: pd.DataFrame, budget: int, seed: int = 42) -> pd.DataFrame:
    budget = int(max(0, budget))
    if budget == 0 or len(df) == 0:
        return df.iloc[0:0].copy()
    return df.sample(n=min(budget, len(df)), random_state=seed).copy()


def select_top(
    df: pd.DataFrame,
    column: str,
    budget: int,
    ascending: bool = False,
) -> pd.DataFrame:
    budget = int(max(0, budget))
    if budget == 0 or len(df) == 0:
        return df.iloc[0:0].copy()
    ranked = df.dropna(subset=[column]).sort_values(column, ascending=ascending)
    return ranked.head(budget).copy()


def select_stratified_bins(
    df: pd.DataFrame,
    column: str,
    num_bins: int,
    budget: int,
    seed: int = 42,
) -> pd.DataFrame:
    budget = int(max(0, budget))
    num_bins = int(max(1, num_bins))
    if budget == 0 or len(df) == 0:
        return df.iloc[0:0].copy()

    work = df.dropna(subset=[column]).copy()
    if len(work) == 0:
        return work

    work["_bin"] = pd.qcut(work[column], q=num_bins, labels=False, duplicates="drop")
    present_bins = [int(b) for b in sorted(work["_bin"].dropna().unique().tolist())]
    if not present_bins:
        return work.iloc[0:0].copy()

    per_bin = budget // len(present_bins)
    remainder = budget % len(present_bins)
    samples = []
    rng = np.random.default_rng(seed)

    for i, b in enumerate(present_bins):
        k = per_bin + (1 if i < remainder else 0)
        if k <= 0:
            continue
        group = work[work["_bin"] == b]
        if len(group) <= k:
            samples.append(group)
            continue
        pick_idx = rng.choice(group.index.to_numpy(), size=k, replace=False)
        samples.append(group.loc[pick_idx])

    if not samples:
        return work.iloc[0:0].copy()
    out = pd.concat(samples, axis=0).drop(columns=["_bin"], errors="ignore")
    return out


def farthest_point_sampling(
    descriptors: np.ndarray,
    budget: int,
    seed: int = 42,
) -> List[int]:
    """Greedy FPS in descriptor space (O(budget * N) distance computations).

    Args:
        descriptors: (N, D) array, should be pre-standardized.
        budget:      Number of points to select.
        seed:        Random seed for initial point.

    Returns:
        List of selected row indices.
    """
    N = descriptors.shape[0]
    budget = min(int(max(0, budget)), N)
    if budget == 0:
        return []

    rng = np.random.RandomState(seed)
    min_dist = np.full(N, np.inf, dtype=np.float64)

    first_idx = int(rng.randint(N))
    selected = [first_idx]
    dists = np.linalg.norm(descriptors.astype(np.float64) - descriptors[first_idx], axis=1)
    np.minimum(min_dist, dists, out=min_dist)

    for _ in range(1, budget):
        next_idx = int(np.argmax(min_dist))
        selected.append(next_idx)
        dists = np.linalg.norm(descriptors.astype(np.float64) - descriptors[next_idx], axis=1)
        np.minimum(min_dist, dists, out=min_dist)
        min_dist[next_idx] = -np.inf

    return selected


def select_fps(
    df: pd.DataFrame,
    descriptors: np.ndarray,
    budget: int,
    seed: int = 42,
) -> pd.DataFrame:
    """Select rows via Farthest Point Sampling in pre-standardized descriptor space.

    Args:
        df:          DataFrame aligned with descriptors (same row order).
        descriptors: (N, D) array — must be standardized before calling.
        budget:      Number of pairs to select.
        seed:        Random seed for FPS initial point.

    Returns:
        Sub-DataFrame of selected rows.
    """
    budget = int(max(0, budget))
    if budget == 0 or len(df) == 0:
        return df.iloc[0:0].copy()
    if len(df) != len(descriptors):
        raise ValueError(
            f"df length ({len(df)}) != descriptors length ({len(descriptors)})"
        )
    idx = farthest_point_sampling(descriptors, budget, seed=seed)
    return df.iloc[idx].copy()


def rank_column(
    df: pd.DataFrame,
    column: str,
    ascending: bool,
    rank_col: Optional[str] = None,
) -> pd.DataFrame:
    rank_col = rank_col or f"selected_rank_{column}"
    out = df.copy()
    out[rank_col] = out[column].rank(method="first", ascending=ascending)
    return out
