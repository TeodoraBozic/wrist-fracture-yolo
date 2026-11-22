# Architecture and research pipeline

## Repository review and scope

The starting repository contained a detailed Serbian README, the research outline
at `docs/docs/experiment_plan.md`, a substantial read-only dataset inspector, and
empty directories for the later pipeline. There were no packaging metadata,
dependency definitions, acquisition tools, tests or experiment configurations.
No files were found under `data/`. Existing work and directory names are retained;
the nested legacy research plan remains at its existing path to preserve references.

The first iteration adds an installable `src` package, selective acquisition,
offline tests and a planning configuration. Inspection and acquisition share
`wrist_fracture.data` but remain separate commands. Future modules will be added
when implemented, rather than creating empty training and evaluation abstractions.
Console entry points are canonical; `scripts/figshare_dataset.py` is a thin wrapper.
There is no need for an environment file: public metadata requires no credentials.

## Acquisition evidence and alternatives

On 2026-11-16, the actual official REST response reported article 14825193,
version 2, DOI `10.6084/m9.figshare.14825193.v2`, CC BY 4.0, with these files:

| Name | Bytes |
| --- | ---: |
| dataset.csv | 1,850,724 |
| folder_structure.zip | 26,304,897 |
| images_part1.zip | 4,046,496,654 |
| images_part2.zip | 4,122,307,169 |
| images_part3.zip | 3,851,047,067 |
| images_part4.zip | 4,207,646,096 |
| Total | 16,255,652,607 |

All six had matching supplied/computed MD5 fields. This is a dated inventory,
not a hardcoded download manifest. The command always resolves filenames through
the API. No dataset payload was downloaded or archive inspected in this iteration;
the contents of `folder_structure.zip` must still be checked before use.

Sources: [official API](https://api.figshare.com/v2/articles/14825193),
[API reference](https://docs.figshare.com/v2/),
[author publication](https://doi.org/10.1038/s41597-022-01328-z), and
[author notebooks](https://figshare.com/articles/software/GRAZPEDWRI-DX_notebooks/19330688).

The official release is downloadable files, not a verified Python streaming dataset
service. The authors provide notebooks, which are usage examples rather than an
established remote-loading API. No official Python remote loader was verified.
A third-party [Hugging Face repackaging](https://huggingface.co/datasets/Layered-Labs/clearwrist-pediatric-wrist-xray)
advertises embedded images and assigned splits. It may offer convenient Python
loading, but equivalence of pixels, labels, patient assignments and release version
has not been established here; it is not the research source of truth.

Direct remote loading saves initial local storage but depends on connectivity,
remote layout, decoding behavior and potentially expensive repeated reads. ZIP
payloads are not automatically efficient random-access training datasets. API
acquisition discovers exact files and source checksums with little extra tooling.
A verified local copy costs disk space but supports repeatable inspection, fixed
patient manifests, efficient repeated training and offline work. Use local verified
originals for the main experiments; never overwrite originals during conversion.

## Downloader guarantees and limits

Downloads are explicit by exact filename, sequential, HTTPS and streamed in 1 MiB
chunks. Size and available source MD5 are verified before a temporary file is
renamed. Existing completed files are checked and reused; mismatches fail without
replacement. Source MD5 validates transfer integrity, not cryptographic authenticity.
SHA-256 is recorded for local identification, and an absent source MD5 is explicitly
marked as unverified beyond size. Full article metadata, file metadata, UTC time,
local path and verification are stored as separate `.provenance/*.json` events.

Transient HTTP/network failures and integrity failures have bounded exponential
retries; permanent HTTP errors fail immediately. Timeout means per-request/read
timeout, not a total transfer deadline. Partial downloads restart rather than resume.
An exclusive per-file lock prevents cooperating writers. After a process crash,
inspect/remove a stale `.NAME.lock` only after checking no download is active; a
stale `.NAME.part` is restarted on the next attempt. Do not edit the output directory
concurrently with acquisition. Archives are never extracted. Metadata inspection
can save a snapshot explicitly and never downloads dataset payloads.

## Next stages and acceptance criteria

1. **Inspect real data.** Explicitly select metadata/annotations first. Inspect
   archive members before controlled extraction to a separate local directory.
   Use the existing inspector to examine headers, bit depth, dimensions, annotation
   schemas, all classes, geometries, patient identifiers and missing/corrupt pairs.
   Review actual CSV rows and annotation overlays. Missing annotations are unknown,
   not negative examples. Retain other annotations and all original files.
2. **Define preprocessing.** Document source coordinate conventions and boundary
   checks before box conversion. Inspect original PNG/TIFF headers before choosing
   decoder, intensity windowing/scaling or channel replication. Any 16-to-8-bit
   conversion is a separate documented derivative with source hashes and visual
   validation; keep the originals. Decide negative inclusion only after reconciling
   CSV tags and box coverage. Do not silently convert other geometries into boxes.
3. **Create patient manifests.** Verify patient IDs against metadata and documented
   naming. Quarantine missing/conflicting identities. Assign patients, not images,
   using a seeded algorithm; assert pairwise-disjoint patient sets and full coverage.
   Freeze validation/test before deriving nested patient-level training fractions.
   Record both intended fraction and achieved patient/image counts. Group repeated
   examinations with the same patient. Check duplicate images across identities.
   Store manifests with hashes and version metadata in `data/splits/`; review privacy
   before committing any identifiers. No split ratios are scientifically finalized.
4. **Prepare YOLO derivatives.** Export verified fracture boxes with an explicit
   source class map to `data/processed/`, generate dataset YAML and validate normalized
   coordinates, image-label correspondence and representative overlays. Test box
   conversion and patient disjointness before training. Keep validation/test identical
   across all capacity and data-size comparisons.
5. **Train with configurations.** Implement config validation and a thin Ultralytics
   runner; finalize YOLO11-S/M/L settings, pretrained policy, seeds, resolution,
   optimizer and fair batch/compute policies. Install and pin compatible PyTorch,
   Ultralytics and CUDA only when hardware is selected. Seed Python, NumPy, PyTorch
   and workers, request deterministic behavior and record nondeterministic operations.
   Use independent seed repetitions for final estimates. The YAML here is a plan,
   not an executable or finalized protocol.
6. **Track every run.** Use unique output directories under `results/`. Save resolved
   configuration, model architecture and checkpoint hash, Git commit and dirty patch,
   dataset DOI/version and source checksums, split/fraction manifest hashes, seed,
   dependency freeze, OS/Python/CUDA/cuDNN/driver/GPU/CPU details and timestamps.
   Pin an exact validated environment for final experiments; current dependency
   ranges are compatibility bounds, not a reproducibility lock.
7. **Evaluate and report.** Save mAP@50, mAP@50:95, precision/recall with explicit
   thresholds and matching rules, training duration, parameter count, FLOPs including
   tool and input shape, inference latency and throughput. Specify warmup, repeat
   counts, batch size, precision, device synchronization and preprocessing/NMS inclusion
   for CPU ONNX and GPU TensorRT benchmarks. Record unavailable FLOPs explicitly.
   Use patient-level resampling for uncertainty; reserve test data for the frozen
   protocol. AUC-ROC requires a defined image/patient scoring rule and clinical target,
   so it is optional despite appearing in the legacy outline. Export publication
   figures/tables from saved metrics; never invent missing results.

Reusable processing, splitting, YOLO export, training, evaluation, tracking and
visualization belong under `src/wrist_fracture/` when implemented. Notebooks call
those modules for exploration and comparisons; scripts only orchestrate entry
points. Tests should cover scientific invariants rather than mirror every function.
