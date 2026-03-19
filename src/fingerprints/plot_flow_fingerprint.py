"""
plot_flow_fingerprint.py
========================
Minimal plotting helpers for flow_fingerprint stats.json.

Functions:
- plot_histograms(stats, out_dir)      # mag, angle, delta, div, curl + joint heatmap
- plot_spatial_maps(stats, out_dir)    # motion_prob + mean_magnitude heatmaps

Optional: plot_overlay(list_of_stats, labels, out_dir) to compare multiple datasets.
"""

from __future__ import annotations
import os, json
import numpy as np
import matplotlib.pyplot as plt

def _load(path):
    with open(path, "r") as f:
        return json.load(f)

def plot_histograms(stats: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    bins = stats["bins"]; H = stats["hists"]

    # 1D helper
    def _plot_1d(edges, p, title, fname):
        centers = 0.5 * (np.array(edges[:-1]) + np.array(edges[1:]))
        plt.figure()
        plt.plot(centers, p, linewidth=2)
        plt.title(title)
        plt.xlabel(title.split()[0])
        plt.ylabel("Probability")
        if "mag" in title.lower() or "delta" in title.lower():
            plt.xscale("log")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname), dpi=160)
        plt.close()

    _plot_1d(bins["mag_edges"],   np.array(H["mag"]),   "Magnitude (px)", "hist_mag.png")
    _plot_1d(bins["ang_edges"],   np.array(H["angle"]), "Angle (rad)",    "hist_angle.png")
    _plot_1d(bins["delta_edges"], np.array(H["delta"]), "Delta (px)",     "hist_delta.png")
    _plot_1d(bins["div_edges"],   np.array(H["div"]),   "Divergence",     "hist_div.png")
    _plot_1d(bins["curl_edges"],  np.array(H["curl"]),  "Curl (z)",       "hist_curl.png")

    # Joint mag-angle as heatmap (mag x angle)
    J = np.array(H["joint_mag_angle"])
    mag_edges = np.array(bins["joint_mag_edges"]); ang_edges = np.array(bins["joint_ang_edges"])
    plt.figure()
    # imshow expects [rows, cols] → [mag_bins, ang_bins]
    plt.imshow(J, aspect="auto", origin="lower",
               extent=[ang_edges[0], ang_edges[-1], mag_edges[0], mag_edges[-1]])
    plt.yscale("log")
    plt.colorbar(label="Probability")
    plt.xlabel("Angle (rad)")
    plt.ylabel("Magnitude (px, log)")
    plt.title("Joint: Magnitude × Angle")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "joint_mag_angle.png"), dpi=160)
    plt.close()

def plot_spatial_maps(stats: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    S = stats["spatial"]
    prob = np.array(S["motion_prob"])
    mean = np.array(S["mean_magnitude"])

    def _heat(img, title, fname, log=False):
        plt.figure()
        if log:
            img = np.log10(np.maximum(img, 1e-6))
        plt.imshow(img, origin="upper")
        cb = plt.colorbar()
        cb.set_label("log10" if log else "value")
        plt.title(title)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname), dpi=160)
        plt.close()

    _heat(prob, "Spatial motion probability P[m>τ]", "spatial_motion_prob.png", log=False)
    _heat(mean, "Spatial mean magnitude E[m]", "spatial_mean_mag.png", log=True)

def plot_overlay(stats_list: list, labels: list, out_dir: str):
    """
    Overlay 1D histograms for multiple datasets for quick comparisons.
    Assumes all have the SAME bin edges (produced by the same config).
    """
    os.makedirs(out_dir, exist_ok=True)
    keys = [("mag", "mag_edges", "Magnitude (px)", "overlay_mag.png", True),
            ("angle", "ang_edges", "Angle (rad)", "overlay_angle.png", False),
            ("delta", "delta_edges", "Delta (px)", "overlay_delta.png", True),
            ("div", "div_edges", "Divergence", "overlay_div.png", False),
            ("curl", "curl_edges", "Curl (z)", "overlay_curl.png", False)]
    for hkey, ekey, title, fname, logx in keys:
        plt.figure()
        for stats, lab in zip(stats_list, labels):
            edges = np.array(stats["bins"][ekey])
            centers = 0.5 * (edges[:-1] + edges[1:])
            p = np.array(stats["hists"][hkey])
            plt.plot(centers, p, label=lab, linewidth=2)
        if logx:
            plt.xscale("log")
        plt.grid(True, alpha=0.3)
        plt.title(title)
        plt.xlabel(title.split()[0])
        plt.ylabel("Probability")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname), dpi=160)
        plt.close()
