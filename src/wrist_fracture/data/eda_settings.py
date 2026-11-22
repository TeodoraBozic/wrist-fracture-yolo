"""Shared notebook inputs and reproducible visual selection."""

from pathlib import Path

import pandas as pd

from .eda import link_annotations, load_metadata, read_yolo_zip
from .image_eda import index_images

DATASET = "GRAZPEDWRI-DX v2"
SEED = 42
USE_CACHE = True
GALLERY_PAGE_SIZE = 6


def project_root():
    for path in (Path.cwd(), *Path.cwd().parents):
        if (path / "pyproject.toml").is_file() and (path / "src/wrist_fracture").is_dir():
            return path
    raise RuntimeError("Pokrenite kernel iz korena projekta ili direktorijuma notebooks/.")


def load_inputs():
    root = project_root()
    raw = root / "data/raw/grazpedwri-dx"
    csv_path, annotation_path = raw / "dataset.csv", raw / "folder_structure.zip"
    metadata = load_metadata(csv_path)
    audit = read_yolo_zip(annotation_path)
    index, issues = index_images([raw / f"images_part{i}.zip" for i in range(1, 5)])
    return root, csv_path, annotation_path, metadata, audit, index, issues


def analytical_images(metadata, audit, index):
    """Use unique physical images and attach only unambiguous metadata."""
    linked = link_annotations(metadata, audit).rename(columns={"member": "annotation_member"})
    unique = linked.loc[~linked.annotation_state.isin(["invalid_identifier", "ambiguous_metadata"])]
    images = index.loc[~index.filestem.duplicated(keep=False)]
    return images.merge(unique, on="filestem", how="left", validate="one_to_one")


def stratified_examples(frame, strata, *, per_group=1, seed=SEED):
    """Select a bounded number per observed stratum, including missing values."""
    if per_group < 1:
        raise ValueError("per_group must be positive")
    ordered = frame.sort_values("image_key").drop_duplicates("image_key")
    groups = ordered.groupby(strata, observed=True, dropna=False)
    result = []
    for values, group in groups:
        if not isinstance(values, tuple):
            values = (values,)
        reason = " | ".join(f"{name}={value if pd.notna(value) else 'nedostaje'}"
                            for name, value in zip(strata, values))
        result.append(group.sample(n=min(per_group, len(group)), random_state=seed)
                      .assign(reason=reason))
    return pd.concat(result, ignore_index=True) if result else ordered.assign(reason="")
