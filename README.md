# SDrAwberry: Scale-Referenced Depth-Anything-3 Long-Sequence Reconstruction for Berry Volume Estimation

At a high level the pipeline is:

1. **Batch multi-camera footage** with `batch_divider_DA3.py`, which flattens each mini-scene, produces consistency summaries, and invokes either the DA3 CLI or a legacy single-scene script.
2. **Reconstruct each batch** via DA3 (default) to obtain COLMAP-formatted sparse models plus optional per-batch point clouds.
3. **Stream-merge batches** with yaw-constrained ICP (`icp_merge.py`) or the diagnostic variant `icp_ploting`, yielding one global point cloud in robot coordinates.
4. **Isolate individual plants** in the merged point cloud using `pcd_cluster_v4.py`, which fits the ground plane, runs HSV-based masking, and clusters vegetation with DBSCAN plus ground-like rejection rules.
5. **Estimate canopy/box volumes** through watertight concave hull meshes produced by `concave_hull.py`, with optional color transfer and scale-aware reporting.

The remainder of this README documents each component in the new workflow.

---

## Repository Contents

| File | Description |
| --- | --- |
| `batch_divider_DA3.py` | Multi-camera batching wrapper. Splits sequences with overlap, renames images, writes JSON/TXT summaries, and runs each batch through either a VGGT single-scene script or the DA3 CLI (recommended). Includes `--batch_range`, DA3 model/quality knobs, and automatic COLMAP export collection. |
| `icp_merge.py` | Streaming point-cloud merger that consumes the per-batch COLMAP folders (`sparse/<id>/points.ply`). Uses COLMAP feature correspondences and yaw-only (rotation around the Y-axis) ICP refinements, voxel hashing, and sliding-window deduplication to produce a global `.ply`. |
| `icp_ploting` | Diagnostic variant of the merger with additional ICP modes (translation-only, matched pairs, yaw-only refined) and automatic logging/plotting of RMSE per adjacent batch pair (`ICP_out/registration_errors.csv` + `.png`). |
| `pcd_cluster_v4.py` | Plant-centric clustering script driven by a parameter block at the top of the file. Performs HSV masking, plane fitting (optionally using only non-plant points), DBSCAN, per-cluster filtering, and visualization/saving of clustered clouds. |
| `concave_hull.py` | Unified watertight concave hull (alpha-shape) workflow with jitter/normalization/fill-hole options, color transfer, and optional global scaling to report physical units. Outputs every tested `k` mesh plus `summary_concave.json`. |
| `README.md` | This document. |

---

## Data Layout

The batching wrapper expects the following tree:

```
scene_dir/
  images/
    <cam_serial_1>/
      rgb/
        rgb_<frame_id>.png
    <cam_serial_2>/
      rgb/
        rgb_<frame_id>.png
    ...
```

* `frame_id` is a decimal string (leading zeros allowed).
* All cameras sit under `images/`, and each contains an `rgb/` subfolder.
* Missing frames per camera are tolerated—the summary will flag counts that deviate from the expected number of cameras.

Each batch is flattened into `scene_dir/batches/<NN>/images/` with files renamed to `<frame_id>_<cam>.png`, so DA3 or any single-scene script sees a standard COLMAP-style folder.

---

## Dependencies & Installation

* **Python 3.9+** with the following packages (install via `pip` or `conda`): `numpy`, `opencv-python`, `open3d`, `matplotlib`, `seaborn`, `pycolmap`, `trimesh` (optional but preferred for PLY export), and `torch` if you still run the VGGT script.
* **Depth-Anything-3 CLI (`da3`)**. Install from Meta's Depth-Anything-3 release or the official PyPI package. The CLI downloads model weights (default `depth-anything/DA3NESTED-GIANT-LARGE-1.1`) on first use.
* **COLMAP model I/O helper** `read_write_model.py` needs to be available on `PYTHONPATH` for `icp_merge.py`/`icp_ploting` (drop the file next to the scripts or install COLMAP).
* A CUDA GPU is required for DA3 reconstruction. `batch_divider_DA3.py` exposes flags to force CPU but it will be extremely slow.

Example environment setup:

```bash
conda create -n straw-da3 python=3.10
conda activate straw-da3
pip install numpy opencv-python open3d matplotlib seaborn pycolmap trimesh
pip install git+https://github.com/DepthAnything/Depth-Anything-V3  # or the official DA3 package
```

Make sure the `da3` executable resolves on your `PATH` before running the wrapper.

---

## Step-by-Step Workflow

### 1. Partition Multi-Camera Sequences & Run DA3

`batch_divider_DA3.py` handles both the flattening of `images/` and the per-batch reconstruction. Key capabilities:

* Stable sort of images per frame by `(cam_serial, filename)` to ensure deterministic multi-view ordering.
* Automatic padding of batch folder names (`01`, `02`, …) with detection/renaming of legacy `001` style directories.
* Summary exports (`batch_summary.json` + `.txt`) that list frames, detected cameras, and missing-frame warnings before any reconstruction starts.
* Flexible selection of batches through `--only_batch`, `--max_batches`, or the new `--batch_range start-end` flag (mutually exclusive).
* DA3 backend options (`--backend da3`) that directly invoke `da3 images ... --export-format colmap` and collect COLMAP BIN results under `scene_dir/sparse/<NN>/`.

Example command (DA3 backend, overlapping batches, automatic PLY export):

```bash
python batch_divider_DA3.py \
  --scene_dir /path/to/scene_dir \
  --batch_size 12 \
  --overlap_frames 6 \
  --symlink \
  --backend da3 \
  --batch_range 1-40 \
  --da3_model_dir depth-anything/DA3NESTED-GIANT-LARGE-1.1 \
  --da3_device cuda \
  --da3_conf_thresh_percentile 35 \
  --da3_num_max_points 1500000 \
  --da3_process_res 576 \
  --da3_export_ply
```

Notes:

* Use `--symlink` to avoid copying large image trees; the wrapper removes/relinks files safely each run.
* `--cams_per_frame` enforces an expected count; otherwise the script infers it from the directory structure and reports frames that deviate.
* With `--backend vggt` the wrapper will run a single-scene Python script (default `demo_colmap_perc.py`) inside each batch and then move its `sparse/` outputs next to the batching root. This mode is retained only for backward compatibility.
* DA3-specific options such as `--da3_ref_view_strategy` and `--da3_disable_triton` are exposed for fine control over quality/speed and compatibility.

Outputs after this step:

* `scene_dir/batches/<NN>/images/` – flattened per-batch folders.
* `scene_dir/sparse/<NN>/` – DA3-exported COLMAP models (`cameras.bin`, `images.bin`, `points3D.bin`, and optionally `points.ply`).
* `scene_dir/batch_summary.{json,txt}` – metadata for quick sanity checks.

### 2. Merge Batch Reconstructions

Run `icp_merge.py` once all batches have finished. Provide the base directory, the `sparse/` subfolder, and the per-batch PLY filename (default `points.ply`). Example:

```bash
python icp_merge.py \
  --base_dir /path/to/scene_dir \
  --sparse_subfolder sparse \
  --ply points.ply \
  --out /path/to/merged_da3.ply \
  --batch_range 1:60 \
  --max_batches 60
```

Highlights:

* Batches are loaded in numeric order; mixed numeric/non-numeric names fall back to lexicographic sorting.
* COLMAP correspondences are built from shared `(image_name, xy)` observations between consecutive batches, producing 3D–3D matched pairs for an Umeyama initialization.
* ICP refinement assumes the robot travels along straight rows; optimization is constrained to **yaw rotation + translation** to improve robustness.
* Point deduplication combines match-based seed removal and sliding-window KD-tree geometry pruning. Remaining points are accumulated via a voxel hash (`VOX` controls spatial resolution).
* `read_write_model.py` must be importable; place it in the repo root if COLMAP is not installed system-wide.

The merged `.ply` will later feed clustering and meshing.

### 3. Cluster Plants With `pcd_cluster_v4.py`

`pcd_cluster_v4.py` is intentionally parameterized through variables at the top of the file—edit them before running:

* `ROOT` & `INPUT_PCD` point to the merged output.
* Set `USE_AUTO_SCALE=True` to derive DBSCAN/plane thresholds from the scene’s bounding-box diagonal, or keep manual millimeter-like thresholds if the point cloud lacks metric scale.
* `Z_AXIS_INVERTED` handles the sign flip introduced by VGGT-style coordinate systems (keep `True` for DA3 exports where plants appear at lower Z).
* Plane fitting can use only non-plant data (`USE_NONPLANT_FOR_PLANE_FIT`) to avoid leaf bias. `USE_NONPLANT_HEIGHT_CUT` is disabled by default to retain sloped ridges.
* HSV ranges isolate leaves vs stems; tune for each dataset. Masks are combined to form a “plant” mask and everything else is optionally used for plane estimation.
* DBSCAN parameters (`DBSCAN_EPS_MANUAL`, `DBSCAN_MIN_PTS`) and size filters (`CLUSTER_MIN_SIZE`, `CLUSTER_MAX_SIZE`) determine which clusters are considered valid plants.
* Ground-like cluster rejection uses median height vs plane, height range, and PCA planarity to remove soil or mulch clusters.
* Output toggles (`SAVE_CLUSTERED_PCD`, `SAVE_EACH_CLUSTER_SEPARATELY`, etc.) control how point clouds are written. Visualization exports color every cluster uniquely for quick QA.

Run the script directly:

```bash
python pcd_cluster_v4.py
```

Interactive Open3D visualizers appear for manual inspection. Adjust thresholds iteratively until each plant forms a clean cluster.

### 4. Build Concave-Hull Meshes and Volumes

Use `concave_hull.py` to turn each plant (or calibration box) cluster into a watertight mesh and compute volume:

```bash
python concave_hull.py \
  --in_pcd /path/to/cluster_07.ply \
  --out_dir /path/to/cluster_07_concave \
  --invert_z \
  --use_scale --scale 0.001 \
  --alpha_voxel 0.0025 \
  --alpha_k_list "2,3,4,5,6,8,10,15" \
  --fill_holes --hole_size_factor 4 \
  --keep_color --color_knn 3
```

Key behaviors:

* Applies z-axis inversion and global scaling consistently before any processing. With `--use_scale` the script assumes points are already in meters and reports volume in cubic centimeters.
* Cleans the cloud (removes NaNs/duplicates) and runs a dedicated alpha-stage voxel downsampling to stabilize the mesh.
* Computes a nearest-neighbor median distance that drives the alpha sweep (`alpha = k * nn_median`).
* Optional jitter, normalization, and Open3D tensor-based hole filling help avoid degenerate tetrahedra or open surfaces.
* Saves every attempted mesh (`*_k*.ply`) and records metadata plus the first watertight mesh that meets the triangle-count threshold in `summary_concave.json`.
* When `--keep_color` is enabled, mesh vertex colors are transferred from the original point cloud via nearest/kNN lookup.

This tool provides a consistent way to compare plant canopies and reference boxes—use the same `alpha_voxel` and `alpha_k_list` across objects for apples-to-apples measurements.

### 5. Optional: ICP Diagnostics & Plots

For sequences where registration quality must be audited, use the plotting variant:

```bash
python icp_ploting \
  --base_dir /path/to/scene_dir \
  --sparse_subfolder sparse \
  --ply points.ply \
  --out /path/to/merged_diagnostics.ply \
  --batch_range 1:80 \
  --icp_mode yaw_pairs_refined
```

`icp_ploting` mirrors the merge logic but additionally:

* Logs per-pair RMSE, correspondence counts, and mode selections in `ICP_out/registration_errors.csv`.
* Generates `ICP_out/icp_alignment_errors.png` (Seaborn bar chart) where each bar corresponds to a batch transition, useful when tuning `MATCH_TO_PLY_RADIUS`, ICP modes, or dedup parameters.

Use this when experimenting with non-default ICP configurations or when diagnosing drift.

---

## Scale & Axis Conventions

* DA3 reconstructions remain **scale ambiguous**. To work in metric units, compute a global scale factor using reference boxes and pass it into `concave_hull.py` via `--use_scale --scale <meters_per_unit>`. The clustering script can also adapt thresholds based on scene size (`USE_AUTO_SCALE`).
* Most exported point clouds use a convention where the robot travels along **+X**, gravity aligns roughly with **-Z**, and the strawberries sit at lower Z than the ground plane. Flip `Z_AXIS_INVERTED` or `--invert_z` if your dataset differs.

---

## Troubleshooting & Tips

* **Missing frames warnings:** `batch_summary.txt` lists frames that do not contain the expected number of camera images. Investigate capture issues if the count stays low; reconstructions will still run but matches may be sparser.
* **DA3 CLI restarts:** The wrapper deletes each batch’s export directory before launching DA3 to avoid interactive prompts. If DA3 crashes, rerun the wrapper with `--only_batch <id>` after fixing the issue.
* **read_write_model errors:** Ensure `read_write_model.py` (from COLMAP) is colocated with the scripts or available on `PYTHONPATH`; otherwise the ICP scripts cannot parse BIN/TXT reconstructions.
* **Plane fit failures:** When color masking leaves too few non-plant points, `pcd_cluster_v4.py` automatically falls back to using all points for RANSAC. Check HSV thresholds or disable `USE_NONPLANT_FOR_PLANE_FIT` if this happens frequently.
* **Concave hull not watertight:** Increase `--hole_size_factor`, widen `--alpha_k_list`, or enable `--normalize`/`--jitter_ratio` slightly. Review each saved `*_k*.ply` to understand where the mesh breaks.
* **Long sequences:** Use `--batch_range` in both the batching script and ICP merge to focus on subsets without touching the rest of the pipeline. Window sizes (`WINDOW_KDT`, `WINDOW_MATCH`) in the merge scripts determine both speed and dedup aggressiveness—decrease them for huge datasets when memory becomes an issue.

This README will continue to evolve alongside the DA3-based workflow. Please update it when new calibration, scaling, or benchmarking modules are added.
