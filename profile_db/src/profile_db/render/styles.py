# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Deterministic style constants for the render layer (DESIGN.md 7).

Every visual decision that could vary between renders lives here: figure
size, DPI, engine colors, line widths, and the downsampling budget. The
Agg backend (selected in ``renderers``) plus this single constant module
is what makes ``render(run, kind, params)`` byte-stable for a fixed
Python/matplotlib pair.
"""

from __future__ import annotations

# Figure geometry (inches) and rasterization.
FIGURE_WIDTH_IN = 8.0
FIGURE_HEIGHT_IN = 4.5
DPI = 100
# Per-image byte budget: a PNG over this size is re-rendered at a lower
# DPI (halved repeatedly) and flagged ``downsampled`` in the manifest.
DEFAULT_IMAGE_MAX_BYTES = 1 * 1024 * 1024
MIN_DPI = 8

# Cache total byte budget (LRU, across all runs and kinds).
DEFAULT_CACHE_MAX_BYTES = 200 * 1024 * 1024

# Engine colors: known engines get fixed colors; unknown ones are assigned
# from the fallback palette in first-seen sorted order (still deterministic
# for a given run because the engine set is fixed). The fallback palette is
# kept disjoint from RESERVED_COLORS below so an unknown engine can never
# be painted with a color that carries another meaning.
ENGINE_COLORS: dict[str, str] = {
    "aic": "#1f77b4",
    "aiv": "#ff7f0e",
}
FALLBACK_PALETTE: tuple[str, ...] = (
    "#00838f",
    "#673ab7",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
)

# Highlight / annotation colors.
TASK_HIGHLIGHT = "#d62728"
READY_LINE = "#2ca02c"
WINDOW_LINE = "#999999"
GAP_FILL = "#ffeb3b"
DEPENDENCY_ARROW = "#444444"

# Identity-bearing colors that task bars and lines must never share with
# the fallback palette: engine-fixed colors plus highlight/ready/gap fills.
RESERVED_COLORS: tuple[str, ...] = (
    *ENGINE_COLORS.values(),
    TASK_HIGHLIGHT,
    READY_LINE,
    GAP_FILL,
)

# Line widths and opacity.
BAR_EDGE_WIDTH = 0.3
BAR_ALPHA = 0.85
SIBLING_ALPHA = 0.4
ARROW_WIDTH = 0.6
ANNOTATION_FONT_SIZE = 9


def color_for_engine(engine: str, ordered_engines: list[str]) -> str:
    """Stable color for an engine: fixed map first, then palette by rank."""
    if engine in ENGINE_COLORS:
        return ENGINE_COLORS[engine]
    index = ordered_engines.index(engine)
    return FALLBACK_PALETTE[index % len(FALLBACK_PALETTE)]
