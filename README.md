# Straw-VGGT: Long-Sequence Multi-Camera 3D Reconstruction for Strawberry Plants

This repository contains an experimental pipeline for reconstructing strawberry plants in the field using the VGGT model.  
The focus is on **long multi-camera sequences** captured by a field robot moving along straight rows, and on **per-plant volume estimation** from 3D reconstructions.

The code here provides:

- Sequence partitioning of long, multi-camera sequences into overlapping batches.
- Batch-wise 3D reconstruction using VGGT, with COLMAP-compatible outputs.
- Stitching of per-batch point clouds using yaw-constrained ICP and streaming deduplication.
- Ground-plane fitting, height-based cropping, and color + DBSCAN clustering to isolate individual strawberry plants.
- Hooks for future **scale recovery** via known-volume boxes and **mesh-based volume estimation** per plant.

Your project outline and research questions are captured and reflected in this README.

---

## 1. Overall Pipeline

Conceptually, the full pipeline is:

1. **Sequence Partitioning (done)**  
   - The raw multi-camera sequence is divided into smaller batches (mini-scenes), with at least 50% frame overlap.  
   - Overlap ensures enough common views for 3D point matching and later stitching.

2. **Batch-wise 3D Reconstruction using VGGT (done)**  
   - Each batch is reconstructed independently with VGGT to produce a partial point cloud.  
   - VGGT outputs are **scale-free** (no absolute metric scale); scale is recovered later using known objects.

3. **Point Cloud Stitching (done)**  
   - Overlapping batches are aligned using 2D feature correspondences from COLMAP mapped into 3D.  
   - ICP is constrained to the robot’s motion model: **one yaw rotation + translation** rather than full 6-DoF.  
   - This leverages the fact that the robot moves along a **straight row**, reducing ambiguity and improving robustness.

4. **Scale Recovery (partially done)**  
   - Boxes of **known volume/shape** are placed in the field of view.  
   - These act as scale references to convert VGGT’s arbitrary units into metric units.  
   - The current repository does not yet contain the full implementation, but the downstream pipeline is designed assuming this step will provide a single global (or local-per-region) scale factor.

5. **Post-processing and Volume Estimation (partially done)**  
   - Fit a ground plane with RANSAC.  
   - Retain points above/below the plane depending on configuration.  
   - Cluster strawberry plants (color-based filtering + DBSCAN).  
   - Planned: for each cluster, reconstruct a **mesh** and compute its volume. Meshes are treated as a finer, more accurate representation than concave hulls (as used in SASP).  
   - Planned: rescale the per-plant mesh volumes using the recovered metric scale.

The code in this repo currently implements the core of steps **1–3** and parts of **5**, while **4** (scale) and the final **mesh-based volume estimation** are still to be integrated.

---

## 2. Repository Structure

Top-level files:

- `batch_divider_v2.py`  
  Multi-camera sequence partitioner and batch runner. Split a long sequence into overlapping batches, rename images into a flat format per batch, and run the single-scene reconstruction script (VGGT) for each batch.

- `demo_colmap_perc.py`  
  Single-scene VGGT demo adapted as the **per-batch reconstruction script**. Given a directory with `images/`, it:
  - Loads a pre-trained VGGT model.
  - Predicts camera poses and depth maps.
  - Converts depth maps into point clouds.
  - Optionally performs BA (bundle adjustment) via COLMAP/pycolmap.
  - Writes COLMAP-format sparse reconstruction and a `points.ply` point cloud.

- `icp_merge.py`  
  Streaming multi-batch point cloud merger with **yaw-only ICP refinement** and windowed deduplication. It assumes the robot travels along a straight line and restricts motion to yaw + translation.

- `icp_pointcloud_merge_batch_v8_se3.py`  
  An alternative/experimental streaming merge script with more modes (full SE(3), translation-only, yaw-only), point-to-plane ICP variants, and similar sliding-window deduplication. The recommended default script is currently `icp_merge.py`.

- `pcd_cluster_v1.py`  
  Post-processing on a merged `.ply` point cloud:
  - Fit ground plane using RANSAC.
  - Height-based cropping.
  - Color-based segmentation (leaf/stem HSV thresholds).
  - DBSCAN clustering to separate individual plants.
  - Save combined and/or per-cluster point clouds and visualize results.

- `README.md`  
  This document.

---

## 3. Data Layout and Assumptions

### 3.1. Multi-Camera Scene Layout

`batch_divider_v2.py` assumes a **multi-camera** directory structure:

```text
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

Where:

- `<cam_serial_x>` is the string identifying each camera (e.g. serial number).
- File names follow `rgb_<frame_id>.png`, where `<frame_id>` is a decimal integer, optionally with leading zeros (e.g. `000123`).
- Different cameras may have **missing frames**; the script logs per-frame image counts and warns when they don’t match the expected number of cameras.

Within each batch, the script:

- Groups frames by `frame_id`.
- Sorts images per frame by `(cam_serial, filename)` to ensure consistent ordering.
- Copies or symlinks images into `scene_dir/batches/<NN>/images/`, renaming to:

```text
<frame_id_str>_<cam_serial>.png
```

### 3.2. Single-Scene Layout for VGGT

`demo_colmap_perc.py` expects a per-scene layout:

```text
scene_dir/
  images/
    *.png
```

When run via `batch_divider_v2.py`, each batch directory is already formatted like this:

```text
scene_dir/
  batches/
    01/
      images/
        000123_camA.png
        000123_camB.png
        ...
    02/
      images/
        ...
```

---

## 4. Dependencies and Environment

This repository is not a standalone VGGT implementation; it wraps the **VGGT model and utilities** from Meta’s VGGT project, plus COLMAP/pycolmap and Open3D.

You will need (at minimum):

- Python 3.8+ (3.9/3.10 recommended).
- GPU with CUDA (VGGT is heavy and assumes GPU).
- Python packages:
  - `torch` (with CUDA), `torchvision` as appropriate for your hardware.
  - `numpy`
  - `trimesh`
  - `pycolmap`
  - `open3d`
  - `opencv-python` (for `cv2`)
  - VGGT codebase and its dependencies:
    - `vggt.models.vggt.VGGT`
    - `vggt.utils.load_fn.load_and_preprocess_images_square`
    - `vggt.utils.pose_enc.pose_encoding_to_extri_intri`
    - `vggt.utils.geometry.unproject_depth_map_to_point_map`
    - `vggt.utils.helper.create_pixel_coordinate_grid`, `randomly_limit_trues`
    - `vggt.dependency.track_predict.predict_tracks`
    - `vggt.dependency.np_to_pycolmap.batch_np_matrix_to_pycolmap` and `_wo_track`
- COLMAP model IO utilities (`read_write_model.py` from COLMAP, or equivalent), placed on the Python path so that:
  - `from read_write_model import read_model` works for the ICP scripts.

VGGT weights are downloaded automatically in `demo_colmap_perc.py` via:

```python
_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
```

So you need network access and valid HuggingFace access for the first run.

---

## 5. Step-by-Step Usage

### 5.1. Prepare a Scene

Organize your raw data into the multi-camera layout described above:

```text
scene_dir/
  images/
    cam01/
      rgb/
        rgb_000001.png
        rgb_000002.png
        ...
    cam02/
      rgb/
        rgb_000001.png
        rgb_000002.png
        ...
```

Check that:

- **Frame IDs** are consistent across cameras wherever possible.
- All images are in PNG format and named `rgb_<id>.png`.

### 5.2. Partition Sequence and Run Per-Batch VGGT

Use `batch_divider_v2.py` to partition the long multi-camera sequence into overlapping batches and run `demo_colmap_perc.py` for each batch:

```bash
python batch_divider_v2.py \
  --scene_dir /path/to/scene_dir \
  --batch_size 10 \
  --overlap_frames 5 \
  --symlink \
  --demo_script demo_colmap_perc.py \
  --demo_args "--use_ba" \
  --cams_per_frame 4
```

Key options:

- `--scene_dir`  
  Root folder containing the `images/` tree shown above.

- `--batch_size`  
  Number of frames per batch.

- `--overlap_frames`  
  Number of overlapping frames between consecutive batches.  
  Must satisfy `batch_size > overlap_frames`. Overlap ~50% is recommended for robust stitching.

- `--symlink`  
  Use symlinks instead of copying images (saves disk and time).

- `--only_batch` / `--max_batches`  
  Restrict processing to a single batch or the first N batches.

- `--demo_script` / `--demo_args`  
  Script and extra CLI arguments for the single-scene reconstruction. `demo_args` is a split string, so you can pass `--demo_args "--use_ba --conf_thres_value 60"`, etc.

- `--cams_per_frame`  
  Expected number of cameras per frame. If not supplied, it is inferred from the directory structure.

Outputs in `scene_dir`:

- `batches/<NN>/images/`  
  Per-batch image folders with renamed files.

- `sparse/<NN>/` (after running the per-batch script)  
  COLMAP sparse reconstruction for each batch:
  - `cameras.bin` / `images.bin` / `points3D.bin` (or `.txt` variants, depending on pycolmap).
  - `points.ply` – the point cloud exported by `demo_colmap_perc.py`.

- `batch_summary.json`, `batch_summary.txt`  
  Summaries of batches, camera counts, frame ranges, and basic sanity checks.

### 5.3. Single-Scene VGGT Reconstruction (Per Batch)

If you want to run `demo_colmap_perc.py` manually (outside the wrapper), use:

```bash
python demo_colmap_perc.py \
  --scene_dir /path/to/single_scene_dir \
  --use_ba
```

Where `/path/to/single_scene_dir` contains an `images/` directory with all PNGs inside.  
The script:

- Loads the VGGT model (bfloat16 or float16 depending on GPU capability).
- Loads and crops images to a square resolution (e.g. 1024), then runs VGGT at 518×518.
- Predicts:
  - Camera intrinsics/extrinsics.
  - Depth maps and confidence maps.
- If `--use_ba`:
  - Uses a VGGSfM-style tracker (`predict_tracks`) to generate 2D tracks.
  - Converts to COLMAP format via `batch_np_matrix_to_pycolmap`.
  - Runs bundle adjustment in pycolmap.
- Else:
  - Uses confidence thresholding (percentile-based) and random subsampling to convert depth maps into a sparse set of high-confidence 3D points.
  - Converts directly to a COLMAP sparse model via `_wo_track`.
- Rescales camera intrinsics back to original image resolution and renames images to original filenames.
- Writes a COLMAP model and `sparse/points.ply`.

### 5.4. Streamed Multi-Batch Point Cloud Merging

Once each batch has its own `sparse/<NN>` folder and `points.ply`, you can **stitch all batches** into a single global point cloud using `icp_merge.py`:

```bash
python icp_merge.py \
  --base_dir /path/to/scene_dir \
  --sparse_subfolder sparse \
  --ply points.ply \
  --out /path/to/output_merged.ply \
  --model_ext ".bin" \
  --max_batches 100 \
  --batch_range 1:100
```

Assumptions and behavior:

- `base_dir/sparse/` contains subfolders named by batch index (e.g. `01`, `02`, …) with COLMAP models plus `points.ply`.
- Batches are sorted numerically when possible.
- For each batch:
  1. Load the local `points.ply` (optionally voxel-downsampled).
  2. If not the first batch, load COLMAP models of the current and a few recent parent batches.
  3. Build 3D–3D correspondences using shared 2D features across COLMAP reconstructions.
  4. Estimate relative pose from current to parent:
     - Initial SE(3) alignment via Umeyama.
     - Refine with **yaw-only** ICP (`yaw_pairs_refined`), optimizing only rotation about the Y-axis + translation.
  5. Chain poses along the sequence: `G_cur = G_prev @ T_rel`.
  6. Deduplicate points via:
     - Match-seed-based local deduplication (before global transform).
     - Geometry-based global deduplication using KD-trees over a sliding window of recent global chunks.
  7. Insert surviving points into a `VoxelHash`, aggregating into voxel centroids and averaged colors.

The final merged point cloud is saved as `out` (e.g. `output_merged.ply`).

Notes:

- `icp_merge.py` is tuned for a robot moving along a row, so **yaw-only** rotation is appropriate.  
  For more general motion, `icp_pointcloud_merge_batch_v8_se3.py` provides more flexible SE(3)/translation-only modes.

### 5.5. Ground Plane Fitting and Plant Clustering

`pcd_cluster_v1.py` operates on a merged point cloud (`.ply`) and performs:

1. **RANSAC Ground Plane Fitting**  
   - Uses `open3d.geometry.PointCloud.segment_plane`.
   - Distance threshold can be specified manually or scaled with the scene’s bounding box.

2. **Height-Based Cropping**  
   - Computes signed distances of all points to the fitted plane.  
   - You can keep either the “higher” or “lower” side w.r.t the plane, and drop the outermost percentage of points (controlled via `HEIGHT_PERCENTILE` and `KEEP_HIGHER_SIDE`).

3. **Color-Based Plant Masking (HSV)**  
   - Convert RGB colors to HSV.  
   - Apply different HSV ranges to detect:
     - Leaves (greenish).
     - Stems/petioles (reddish/yellow-brown).  
   - Combined **plant mask** is the union of leaf + stem masks.

4. **DBSCAN Clustering per Plant**  
   - Run DBSCAN on plant-only points:
     - `eps` set manually (`DBSCAN_EPS_MANUAL`) or automatically scaled from scene size.
     - `min_points` set via `DBSCAN_MIN_PTS`.  
   - Each cluster is interpreted as a **single strawberry plant**.

5. **Saving Clustered Point Clouds**  
   - Optionally save **all clusters combined** in one file (`CLUSTERED_OUTPUT_PATH`).  
   - Optionally save **each cluster separately** (`clusters_out/cluster_<id>.ply`).

6. **Visualization**  
   - Visualization 1: cropped cloud with background gray and each plant cluster colored differently.  
   - Visualization 2: plant clusters only, in original RGB colors.

You must edit the top of `pcd_cluster_v1.py` to set:

```python
INPUT_PCD = "/path/to/merged_output.ply"
USE_AUTO_SCALE = True or False
...
```

Then run:

```bash
python pcd_cluster_v1.py
```

Open3D windows will pop up for interactive inspection.

---

## 6. Scale Recovery (Planned / Partially Done)

VGGT reconstructions are **scale ambiguous**: distances are only defined up to a global factor.  
Your current design for scale recovery is:

- Place **boxes of known volume/shape** within the field-of-view.  
- After the global point cloud is merged (step 3), **identify the reconstructed boxes** in the point cloud.
- Fit a geometric model (e.g. bounding box or mesh) to these reconstructed boxes to estimate their volume/size in VGGT units.
- Compute a global scale factor:

```text
scale_factor = (true_box_volume_in_m^3) / (reconstructed_box_volume_in_vggt_units^3)
```

or equivalently in linear dimensions.

This repository does not yet implement:

- Automatic detection/segmentation of the reference boxes.
- The actual computation of the scale factor and application to all points.

The planned approach is analogous to what was done for SASP using known-shape boxes, but adapted to VGGT point clouds and (eventually) meshes.

---

## 7. Mesh Reconstruction and Volume Estimation (Planned)

For accurate per-plant volume estimation, you plan to:

1. Take each plant cluster (point cloud) from `pcd_cluster_v1.py`.
2. Reconstruct a **mesh** (e.g. Poisson surface reconstruction or ball-pivoting) that captures canopy geometry more faithfully than a concave hull.
3. Compute the **mesh volume** in VGGT units.
4. Apply the **scale factor** from the reference boxes to convert to physical volume (e.g. cubic meters or liters).

Comparison to SASP:

- SASP used a concave hull to approximate canopy volume.
- Here you intend to use **meshes**, which provide:
  - smoother, more detailed surface representation,
  - better handling of complex plant shapes,
  - more reliable volume estimates.

Implementation of meshing and scaling is currently **to be done**; the existing code already provides:

- Plant-level point clouds (clusters).
- Visualization and basic clustering diagnostics, making it easier to debug meshing algorithms once implemented.

---

## 8. Research Concerns and Open Questions

### 8.1. Lack of Ground-Truth Plant Volume

One major challenge is that **ground-truth 3D volume** for field-grown strawberry plants is essentially unavailable:

- Apart from known-volume boxes, there is no direct physical measurement to supervise or validate the 3D volume estimates.
- You plan to:
  - Reuse strategies from SASP (e.g. relying on these known-shape boxes for indirect validation).
  - Potentially design controlled experiments (e.g. lab setups) where destructive measurement is possible for a subset of plants.

The current codebase is thus **self-consistent** but not strongly supervised by ground-truth volumes.  
Future work:

- Statistical analysis of volume stability under repeated scans.
- Comparison against proxy measurements (e.g. manual plant height, canopy width, or biomass).

### 8.2. Benchmarking Reconstruction Quality

At present, the most realistic benchmarking target is the **reconstruction component**:

- How well does VGGT reconstruct geometry in this specific field setting (long sequence + multi-camera + outdoor conditions)?
- How does it compare to alternative pipelines (e.g. COLMAP-only, multiview stereo baselines, or other learned methods)?

Open issues:

- Identifying appropriate baselines and metrics:
  - Reprojection error / track consistency.
  - Geometric consistency across overlapping batches.
  - Stability of structure across repeated passes of the robot.
- Designing **credible benchmarks** that show VGGT’s strengths in:
  - Handling long sequences.
  - Exploiting multi-camera setups.
  - Robustness to outdoor lighting and partial occlusions.

You plan to spend more time on this in upcoming iterations, likely by:

- Creating synthetic or semi-controlled scenes with known geometry.
- Leveraging reference objects beyond boxes (e.g. calibration spheres or known structures).

---

## 9. Practical Tips and Gotchas

- Many scripts contain **hard-coded absolute paths** in the defaults (e.g. `BASE_DIR`, `INPUT_PCD`, `OUT_PLY`).  
  Always override these via CLI or edit them to match your environment.

- For `icp_merge.py` and `icp_pointcloud_merge_batch_v8_se3.py`, ensure `read_write_model.py` is importable.  
  You can drop the COLMAP-provided `read_write_model.py` into this directory or add it to `PYTHONPATH`.

- `demo_colmap_perc.py` downloads VGGT weights at first use; if you’re on a cluster or offline environment, download them in advance and load from a local path.

- The ICP merge assumes **yaw-only rotation** around the Y-axis (up). If your robot deviates from a straight line or has significant roll/pitch, you may need the more general SE(3) options in `icp_pointcloud_merge_batch_v8_se3.py` or a different registration strategy.

- Color-based thresholds in `pcd_cluster_v1.py` are tuned for a particular imaging setup.  
  You will likely need to adjust HSV ranges and DBSCAN parameters for different cameras, lighting, or cultivars.

---

## 10. Status Summary

Implementation status relative to the planned pipeline:

- [x] Sequence partitioning of multi-camera sequences (`batch_divider_v2.py`).
- [x] Batch-wise VGGT reconstruction and COLMAP export (`demo_colmap_perc.py`).
- [x] Multi-batch stitching with yaw-constrained ICP and voxel-hash merge (`icp_merge.py` / `icp_pointcloud_merge_batch_v8_se3.py`).
- [x] Ground-plane fitting, height cropping, and plant clustering (`pcd_cluster_v1.py`).
- [ ] Robust scale recovery using known-volume boxes (design complete, implementation pending).
- [ ] Mesh reconstruction per plant and mesh-based volume estimation.
- [ ] Systematic benchmarking of VGGT vs alternatives in long-sequence, multi-camera field environments.

This README will evolve as you add scale recovery, meshing, and benchmarking scripts. If you’d like, the next steps could be to design a small API for per-plant mesh reconstruction and integrate a simple mesh volume estimator into this pipeline.
