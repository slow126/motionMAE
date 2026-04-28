#!/usr/bin/env python3
"""Render slide equation assets with LaTeX and export transparent PNGs."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TEAL = "00B4D8"


def run(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def render_latex_png(name: str, latex_body: str, density: int = 300) -> None:
    with tempfile.TemporaryDirectory(prefix="eq_render_") as tmp:
        tmpdir = Path(tmp)
        tex_path = tmpdir / f"{name}.tex"
        pdf_path = tmpdir / f"{name}.pdf"
        out_base = tmpdir / name

        tex_path.write_text(
            rf"""\documentclass[border=6pt]{{standalone}}
\usepackage{{amsmath,amssymb}}
\usepackage[svgnames]{{xcolor}}
\begin{{document}}
\color[HTML]{{{TEAL}}}
{latex_body}
\end{{document}}
""",
            encoding="utf-8",
        )

        run(
            [
                "pdflatex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                tex_path.name,
            ],
            cwd=tmpdir,
        )
        run(
            [
                "pdftocairo",
                "-png",
                "-transp",
                "-r",
                str(density),
                pdf_path.name,
                out_base.name,
            ],
            cwd=tmpdir,
        )

        png_src = tmpdir / f"{name}-1.png"
        png_dst = OUT_DIR / f"{name}.png"
        png_dst.write_bytes(png_src.read_bytes())


def main() -> None:
    render_latex_png(
        "equation_support_inclusion",
        r"\textbf{$\varepsilon$-Support Inclusion:}\quad "
        r"$S_{\varepsilon}(A \mid B)"
        r"=\frac{1}{\lvert A \rvert}\sum_{a \in A}\mathbf{1}\!\left["
        r"\min_{b \in B}\lVert a-b\rVert_2 \leq \varepsilon\right]$",
        density=360,
    )


if __name__ == "__main__":
    main()
