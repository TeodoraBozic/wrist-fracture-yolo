"""Read-only inspection of an extracted GRAZPEDWRI-DX dataset."""

import argparse
import csv
import json
import math
import re
import statistics
import sys
import warnings
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
FILENAME_PATTERN = re.compile(r"^(\d{4})_\d+_\d+_WRI-[A-Za-z0-9]+_[FM]\d+$")


def image_key(value):
    name = str(value).replace("\\", "/").rsplit("/", 1)[-1]
    if name.lower().endswith((".json", ".xml", ".txt")):
        name = name.rsplit(".", 1)[0]
    if Path(name).suffix.lower() in IMAGE_EXTENSIONS:
        name = name.rsplit(".", 1)[0]
    return name


def boolean_tag(value):
    value = str(value).strip().lower()
    if value in {"1", "1.0", "true", "yes", "positive"}:
        return True
    if value in {"0", "0.0", "false", "no", "negative"}:
        return False
    return None


def describe(values):
    if not values:
        return "unavailable"
    ordered = sorted(values)
    return (f"n={len(values)}, min={ordered[0]:.4g}, "
            f"median={statistics.median(ordered):.4g}, "
            f"p90={ordered[math.ceil(0.9 * len(ordered)) - 1]:.4g}, "
            f"max={ordered[-1]:.4g}")


def print_counts(title, counts, limit=20):
    print(f"\n{title}")
    for label, count in counts.most_common(limit):
        print(f"  {label}: {count}")
    if not counts:
        print("  unavailable")
    if len(counts) > limit:
        print(f"  ... {len(counts) - limit} more categories")


def read_image(path):
    from PIL import Image

    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
            depth = "unknown"
            if image.format == "PNG":
                with path.open("rb") as stream:
                    header = stream.read(29)
                depth = header[24]
            elif image.format == "TIFF":
                depth = image.tag_v2.get(258, "unknown")
            elif image.format in {"JPEG", "BMP"} and image.mode in {"L", "RGB", "RGBA", "CMYK"}:
                depth = 8
            return image.size, (image.format, image.mode, str(depth)), getattr(image, "n_frames", 1)


def read_annotation(path, annotation_format, size, class_names):
    objects = []
    declared_size = None
    tag = None
    if annotation_format == "supervisely":
        content = json.loads(path.read_text(encoding="utf-8-sig"))
        declared_size = (int(content["size"]["width"]), int(content["size"]["height"]))
        for item in content["objects"]:
            geometry = item["geometryType"]
            box = None
            if geometry == "rectangle":
                points = item["points"]["exterior"]
                if len(points) != 2 or any(len(point) != 2 for point in points):
                    raise ValueError("rectangle must have two 2D points")
                box = (*points[0], *points[1])
            elif geometry in {"polygon", "line", "point"}:
                points = item["points"]["exterior"]
                minimum = {"polygon": 3, "line": 2, "point": 1}[geometry]
                if len(points) < minimum or any(
                    len(point) != 2 or not all(math.isfinite(float(value)) for value in point)
                    or not (0 <= float(point[0]) <= size[0] and 0 <= float(point[1]) <= size[1])
                    for point in points
                ):
                    raise ValueError(f"invalid {geometry} points")
            else:
                raise ValueError(f"unsupported geometry: {geometry}")
            objects.append((item["classTitle"], geometry, box))
        tags = {item["name"] for item in content.get("tags", [])}
        tag = True if "fracture" in tags else None
    elif annotation_format == "voc":
        root = ET.parse(path).getroot()
        if root.tag != "annotation":
            raise ValueError("not a VOC annotation")
        declared_size = (int(root.findtext("size/width")), int(root.findtext("size/height")))
        filename = root.findtext("filename")
        if filename and image_key(filename) != image_key(path.name):
            raise ValueError("VOC filename differs from annotation filename")
        for item in root.findall("object"):
            bounds = item.find("bndbox")
            box = None if bounds is None else tuple(
                float(bounds.findtext(field)) for field in ("xmin", "ymin", "xmax", "ymax")
            )
            objects.append((item.findtext("name"), "rectangle" if box else "other", box))
    else:
        for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(f"line {line_number}: expected YOLO detection row with 5 fields")
            class_id = int(fields[0])
            if not 0 <= class_id < len(class_names):
                raise ValueError(f"line {line_number}: class ID outside supplied class list")
            center_x, center_y, width, height = map(float, fields[1:])
            if not all(math.isfinite(value) and 0 <= value <= 1 for value in (center_x, center_y, width, height)):
                raise ValueError(f"line {line_number}: invalid normalized coordinates")
            image_width, image_height = size
            box = ((center_x - width / 2) * image_width, (center_y - height / 2) * image_height,
                   (center_x + width / 2) * image_width, (center_y + height / 2) * image_height)
            objects.append((class_names[class_id], "rectangle", box))
    return objects, declared_size, tag


def inspect(args):
    root = args.dataset.resolve()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    issues = defaultdict(list)
    print(f"Dataset: {root}\nFiles: {len(files)}")
    print_counts("Files by extension", Counter(path.suffix.lower() or "[no extension]" for path in files))
    print_counts("Folder organization (files per relative directory)", Counter(str(path.parent.relative_to(root)) for path in files))
    images = defaultdict(list)
    for path in files:
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            images[image_key(path.name)].append(path)
    print(f"\nImage files: {sum(map(len, images.values()))}; unique filename stems: {len(images)}")
    for key, paths in images.items():
        if len(paths) > 1:
            issues["Ambiguous image stem (excluded)"].append(f"{key}: {len(paths)} files")
    if not images:
        print("No supported images found. Dataset inspection was not performed; extract the dataset first.")
        return 2

    dimensions = {}
    formats = Counter()
    for key, paths in images.items():
        if len(paths) != 1:
            continue
        try:
            size, image_format, frames = read_image(paths[0])
            dimensions[key] = size
            formats[image_format] += 1
            if frames != 1:
                issues["Multi-frame images (only first frame inspected)"].append(str(paths[0]))
        except Exception as error:
            issues["Unreadable or invalid image"].append(f"{paths[0]}: {error}")
    print_counts("Decoded resolutions (width, height)", Counter(dimensions.values()))
    print_counts("Decoded format, mode, stored bits per channel/sample", formats)

    csv_paths = [args.metadata] if args.metadata else [path for path in files if path.name.lower() == "dataset.csv"]
    metadata = {}
    columns = []
    if len(csv_paths) == 1:
        try:
            with csv_paths[0].open(encoding="utf-8-sig", newline="") as stream:
                sample = stream.read(8192)
                stream.seek(0)
                try:
                    dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
                except csv.Error:
                    dialect = csv.excel
                reader = csv.DictReader(stream, dialect=dialect)
                columns = reader.fieldnames or []
                print(f"\nMetadata: {csv_paths[0]}\nColumns: {', '.join(columns)}")
                filename_column = args.filename_column or next((name for name in ("filestem", "filename", "file_name") if name in columns), None)
                if filename_column not in columns:
                    raise ValueError("filename column not found; supply --filename-column")
                rows = defaultdict(list)
                for row in reader:
                    if None in row or any(value is None for value in row.values()):
                        issues["Malformed metadata row"].append(str(reader.line_num))
                        continue
                    key = image_key(row[filename_column].strip())
                    if not key:
                        issues["Empty metadata filename"].append(str(reader.line_num))
                        continue
                    rows[key].append(row)
                for key, entries in rows.items():
                    if len(entries) == 1:
                        metadata[key] = entries[0]
                    else:
                        issues["Duplicate metadata key (excluded)"].append(key)
                for key in metadata.keys() - images.keys():
                    issues["Metadata references missing image"].append(key)
        except (OSError, UnicodeError, csv.Error, ValueError) as error:
            issues["Metadata unavailable"].append(str(error))
    else:
        issues["Metadata unavailable"].append(f"found {len(csv_paths)} dataset.csv files; use --metadata if needed")

    patients = set()
    studies = set()
    identity_sources = Counter()
    for key in images:
        row = metadata.get(key, {})
        match = FILENAME_PATTERN.fullmatch(key)
        filename_patient = match.group(1) if match else None
        patient = row.get(args.patient_column, "").strip()
        if patient.lower() in {"nan", "none", "null", "na"}:
            patient = ""
        if patient and patient.isdigit():
            patient = patient.zfill(4)
        if patient and filename_patient and patient != filename_patient:
            issues["Patient ID conflict (excluded from identity counts)"].append(key)
            continue
        source = "CSV" if patient else "documented filename pattern" if filename_patient else "unknown"
        identity_sources[source] += 1
        patient = patient or filename_patient
        if not patient:
            issues["Unknown patient identity"].append(key)
        if patient:
            patients.add(patient)
            if args.study_column and row.get(args.study_column, "").strip():
                studies.add((patient, row[args.study_column].strip()))
            elif args.study_column:
                issues["Missing study ID"].append(key)
    print(f"\nUnique identified patients: {len(patients)} (coverage below; conflicts excluded)")
    print_counts("Patient identity source per image stem", identity_sources)
    if args.patient_column not in columns:
        issues["Patient metadata column missing; filename fallback only"].append(args.patient_column)
    if args.study_column and args.study_column in columns:
        print(f"Unique (patient, {args.study_column}) pairs: {len(studies)}; confirm this column denotes an examination.")
    else:
        print("Study count: unavailable; supply a verified --study-column. Filename tokens are not assumed to identify examinations.")

    annotation_root = args.annotations.resolve() if args.annotations else root
    annotation_files = sorted(path for path in annotation_root.rglob("*") if path.is_file())
    candidates = {
        "supervisely": [path for path in annotation_files if path.suffix.lower() == ".json" and path.name.lower() != "meta.json" and (args.annotations or "ann" in path.parts or "supervisely" in path.parts)],
        "voc": [path for path in annotation_files if path.suffix.lower() == ".xml"],
        "yolo": [path for path in annotation_files if path.suffix.lower() == ".txt" and path.name.lower() not in {"classes.txt", "obj.names"} and (args.annotations or "labels" in path.parts or "yolov5" in path.parts)],
    }
    print_counts("Annotation format candidates (not combined)", Counter({name: len(paths) for name, paths in candidates.items()}))
    selected = args.annotation_format
    if selected == "auto":
        selected = next((name for name in candidates if candidates[name]), "supervisely")
    print(f"Selected annotation format: {selected}")
    annotations = defaultdict(list)
    for path in candidates[selected]:
        annotations[image_key(path.name)].append(path)
    for key in annotations.keys() - images.keys():
        issues["Annotation references missing image"].append(key)
    class_names = args.classes or []
    if selected == "yolo" and not class_names:
        issues["YOLO class mapping missing"].append("Supply --classes in the exact order from the export metadata; IDs are never guessed.")

    classes = Counter()
    geometries = Counter()
    boxes_per_image = []
    objects_per_image = []
    box_sizes = defaultdict(list)
    fracture_status = {}
    annotation_tags = {}
    for key in images:
        paths = annotations.get(key, [])
        if len(paths) != 1:
            issues["Missing annotation" if not paths else "Duplicate annotation (excluded)"].append(key)
            continue
        if key not in dimensions or (selected == "yolo" and not class_names):
            continue
        width, height = dimensions[key]
        try:
            objects, declared_size, tag = read_annotation(paths[0], selected, (width, height), class_names)
            if declared_size is not None and declared_size != (width, height):
                raise ValueError(f"annotation size {declared_size} differs from image {(width, height)}")
            valid = True
            valid_boxes = 0
            fracture_boxes = 0
            image_classes = Counter()
            image_geometries = Counter()
            image_box_sizes = defaultdict(list)
            for class_name, geometry, box in objects:
                if not isinstance(class_name, str) or not class_name.strip():
                    raise ValueError("missing class name")
                image_classes[class_name] += 1
                image_geometries[(class_name, geometry)] += 1
                if box is None:
                    if class_name == "fracture":
                        valid = False
                        issues["Fracture object without rectangle"].append(key)
                    continue
                left, top, right, bottom = map(float, box)
                if not (all(math.isfinite(value) for value in (left, top, right, bottom))
                        and 0 <= left < right <= width and 0 <= top < bottom <= height):
                    valid = False
                    issues["Invalid or out-of-image bounding box"].append(f"{key}: {class_name} {box}")
                    continue
                valid_boxes += 1
                fracture_boxes += class_name == "fracture"
                box_width, box_height = right - left, bottom - top
                image_box_sizes["width_px"].append(box_width)
                image_box_sizes["height_px"].append(box_height)
                image_box_sizes["area_px"].append(box_width * box_height)
                image_box_sizes["relative_area"].append(box_width * box_height / (width * height))
            classes.update(image_classes)
            geometries.update(image_geometries)
            for metric, values in image_box_sizes.items():
                box_sizes[metric].extend(values)
            objects_per_image.append(len(objects))
            boxes_per_image.append(valid_boxes)
            if valid:
                fracture_status[key] = bool(fracture_boxes)
            annotation_tags[key] = tag
        except (OSError, UnicodeError, ValueError, TypeError, KeyError, IndexError, AttributeError, ET.ParseError) as error:
            issues["Invalid annotation"].append(f"{paths[0]}: {error}")

    print_counts("Named objects by class (selected export; includes invalid boxes)", classes, len(classes))
    print_counts("Objects by class and geometry", geometries, len(geometries))
    print(f"\nObjects per parsed image: {describe(objects_per_image)}")
    print(f"Valid boxes per parsed image: {describe(boxes_per_image)}")
    print_counts("Distribution: objects per parsed image", Counter(objects_per_image))
    for metric, values in box_sizes.items():
        print(f"Bounding boxes, {metric}: {describe(values)}")
    print(f"\nImages with valid fracture box: {sum(fracture_status.values())}")
    print(f"Images with valid annotation but no fracture box: {sum(not value for value in fracture_status.values())}")
    print(f"Images with unknown annotation status: {len(images) - len(fracture_status)}")
    print("No fracture box does NOT establish a negative diagnosis.")

    comparisons = Counter()
    for key in images:
        row = metadata.get(key)
        if row is None:
            issues["Image without unique metadata row"].append(key)
        metadata_tag = boolean_tag(row.get(args.fracture_column, "")) if row else None
        if row is not None and args.fracture_column in columns and metadata_tag is None:
            issues["Unknown CSV fracture tag"].append(f"{key}: {row.get(args.fracture_column)!r}")
        status = fracture_status.get(key)
        comparisons[(str(metadata_tag), str(status))] += 1
        if metadata_tag is not None and status is not None and metadata_tag != status:
            issues["CSV fracture tag / bounding box disagreement"].append(key)
        if annotation_tags.get(key) is True and status is False:
            issues["Supervisely fracture tag without fracture box"].append(key)
        if annotation_tags.get(key) is True and metadata_tag is False:
            issues["Supervisely / CSV fracture tag disagreement"].append(key)
    print_counts(f"CSV '{args.fracture_column}' vs fracture box (True/False/None=unknown)", comparisons)
    if args.fracture_column not in columns:
        issues["Fracture metadata column missing"].append(args.fracture_column)
    print("\nIssues / limitations (counts and up to 5 examples each)")
    for category, examples in issues.items():
        print(f"  {category}: {len(examples)}")
        for example in examples[:5]:
            print(f"    {example}")
    print("\nInspection finished. No files were changed. Review coverage and issues before using any counts.")
    return 1 if issues else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Root of the extracted original dataset")
    parser.add_argument("--metadata", type=Path, help="Explicit dataset CSV path")
    parser.add_argument("--filename-column", help="CSV filename/stem column (auto: filestem, filename, file_name)")
    parser.add_argument("--patient-column", default="patient_id")
    parser.add_argument("--study-column", help="Verified examination ID column; counted together with patient ID")
    parser.add_argument("--fracture-column", default="fracture")
    parser.add_argument("--annotations", type=Path, help="Limit inspection to one annotation export directory")
    parser.add_argument("--annotation-format", choices=("auto", "supervisely", "voc", "yolo"), default="auto")
    parser.add_argument("--classes", nargs="+", help="YOLO class names ordered by numeric ID, starting at zero")
    args = parser.parse_args()
    if not args.dataset.is_dir():
        parser.error(f"dataset directory does not exist: {args.dataset}")
    if args.annotations and not args.annotations.is_dir():
        parser.error(f"annotation directory does not exist: {args.annotations}")
    if args.classes and (len(set(args.classes)) != len(args.classes) or "fracture" not in args.classes):
        parser.error("--classes must contain unique names including the exact class 'fracture'")
    try:
        import PIL
    except ImportError:
        parser.error("Pillow is required to decode and validate images. Install it manually: python -m pip install Pillow")
    try:
        return inspect(args)
    except OSError as error:
        print(f"Inspection could not finish: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
