"""Consistent grayscale galleries and original annotation overlays."""

from __future__ import annotations

import math
import textwrap
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import pandas as pd

from wrist_fracture.data.eda_settings import GALLERY_PAGE_SIZE
from wrist_fracture.data.image_eda import DECODE_ERRORS, ImageStore, box_to_pixels, display_limits


def image_gallery(
    selected: pd.DataFrame, boxes: pd.DataFrame, classes: dict[int, str], *,
    title: str, overlay: str = "none", columns: int = 4,
) -> Any:
    """Render up to 12 originals with optional all/fracture box overlays.

    P1/P99 limits affect rendering only; pixel measurements always use raw arrays.
    Image extent uses continuous edges (0..width, 0..height), matching YOLO boxes.
    """
    if overlay not in {"none", "all", "fracture"}:
        raise ValueError("overlay must be none, all or fracture")
    if selected.empty:
        return None
    selected = selected.head(12)
    rows = math.ceil(len(selected) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(3.5 * columns, 4.2 * rows),
                             squeeze=False)
    colors = plt.get_cmap("tab10")
    used = set()
    with ImageStore() as store:
        for ax, row in zip(axes.flat, selected.to_dict("records")):
            ax.axis("off")
            try:
                array, info = store.decode(row)
                if array.ndim != 2 or info["mode"] in {"P", "1"}:
                    ax.text(.5, .5, f"Nije grayscale za prikaz: {info['mode']}",
                            ha="center", transform=ax.transAxes)
                    continue
                low, high = display_limits(array)
                ax.imshow(array, cmap="gray", vmin=low, vmax=high,
                          extent=(0, info["width_px"], info["height_px"], 0))
                if overlay != "none":
                    relevant = boxes.loc[boxes.filestem.eq(row["filestem"]) & boxes.valid.eq(True)]
                    if overlay == "fracture":
                        relevant = relevant.loc[relevant.class_name.eq("fracture")]
                    for box in relevant.itertuples():
                        x0, y0, x1, y1 = box_to_pixels(
                            box.cx, box.cy, box.width, box.height,
                            info["width_px"], info["height_px"],
                        )
                        color = colors(int(box.class_id) % 10)
                        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                                               fill=False, edgecolor=color, linewidth=1.5))
                        ax.text(x0, y0, box.class_name, fontsize=7, color="white",
                                bbox={"facecolor": "black", "alpha": .75, "pad": 1})
                        used.add(int(box.class_id))
                reason = textwrap.fill(str(row.get("reason", "")), width=34)
                context = (f" | fracture box={row.get('fracture_boxes', '?')}"
                           if overlay != "none" else "")
                ax.set_title(f"{row['filestem']}\nproj={row.get('projection', '?')} | "
                             f"starost={row.get('age', '?')}{context}\n{reason}", fontsize=8)
            except DECODE_ERRORS as error:
                ax.text(.05, .5, f"Prikaz nije uspeo:\n{error}", fontsize=7,
                        transform=ax.transAxes, wrap=True)
    for ax in list(axes.flat)[len(selected):]:
        ax.axis("off")
    fig.suptitle(title + "\nP1–P99 mapiranje samo za prikaz; izvori su nepromenjeni",
                 fontsize=11)
    if used:
        handles = [Patch(facecolor=colors(i % 10), label=classes[i]) for i in sorted(used)]
        fig.legend(handles=handles,
                   loc="lower center", ncol=min(len(used), 5), fontsize=8)
    fig.tight_layout(rect=(0, .04 if used else 0, 1, .94))
    return fig


def show_gallery_pages(selected, boxes, classes, *, title, overlay="none"):
    """Display every selected row, with a fixed maximum per figure."""
    for start in range(0, len(selected), GALLERY_PAGE_SIZE):
        page = selected.iloc[start:start + GALLERY_PAGE_SIZE]
        fig = image_gallery(page, boxes, classes, title=title, overlay=overlay, columns=3)
        plt.show()
        plt.close(fig)
