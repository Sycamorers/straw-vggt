#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
concave_hull_volume_v3.py

Goal:
- Use ONE consistent method (concave hull / alpha shape) to produce a WATERTIGHT mesh
  and compute volume (Open3D mesh.get_volume()) for both:
  - extrude-filled "box" point clouds
  - plant point clouds

Key engineering fixes (to avoid invalid tetra + non-watertight):
1) Use a dedicated (coarser) alpha-stage voxel downsample: --alpha_voxel
2) Remove duplicated / non-finite points before alpha
3) Optional tiny jitter to break coplanar/grid degeneracy: --jitter_ratio
4) Optional coordinate normalization before alpha for numeric stability: --normalize
5) Optional mesh hole filling (tensor fill_holes): --fill_holes + --hole_size_factor
6) Keep largest connected component to remove stray pieces

Outputs:
- concave_alpha_k*.ply for each attempt
- concave_alpha_k*_repaired.ply if hole fill enabled
- summary_concave.json

Notes:
- Mesh.get_volume() requires watertight mesh. If none found, volume is null in summary.
- If you want a guaranteed volume even when concave hull fails, add a voxel fallback separately.

Unit convention added:
- If --use_scale is enabled, we assume points are scaled into meters (m).
  Then volume from Open3D is in m^3, and we report volume in cm^3 (multiply by 1e6).
"""

import os
import json
import argparse
import numpy as np
import open3d as o3d


# ---------------------------- Args ----------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_pcd", required=True, help="Input point cloud (.ply/.pcd)")
    ap.add_argument("--out_dir", required=True, help="Output directory")

    # VGGT corrections
    ap.add_argument("--invert_z", action="store_true", help="Invert z *= -1 before any processing")
    ap.add_argument("--use_scale", action="store_true", help="Apply global scale factor")
    ap.add_argument("--scale", type=float, default=1.0)

    # Preprocess (for reference stats / optional cleaning)
    ap.add_argument("--remove_outlier", action="store_true")
    ap.add_argument("--outlier_nb", type=int, default=30)
    ap.add_argument("--outlier_std", type=float, default=1.8)

    # Alpha-stage downsample (IMPORTANT)
    ap.add_argument("--alpha_voxel", type=float, default=0.002288,
                    help="Voxel size used ONLY for alpha/concave hull stage. "
                         "Use the same value for box and plants to keep consistency.")
    ap.add_argument("--alpha_nn_sample", type=int, default=5000,
                    help="Subsample count to estimate NN median (speed).")

    # Alpha sweep
    ap.add_argument("--alpha_k_list", type=str,
                    default="20,30,40,60,80,100,120,150",
                    help="Comma-separated multipliers. alpha = k * nn_median(alpha_pcd)")
    ap.add_argument("--min_triangles", type=int, default=2000)

    # Degeneracy fixes
    ap.add_argument("--normalize", action="store_true",
                    help="Normalize coordinates (center + scale by bbox diagonal) before alpha, then denormalize mesh back.")
    ap.add_argument("--jitter_ratio", type=float, default=0.001,
                    help="Gaussian jitter sigma = jitter_ratio * nn_median(alpha_pcd). 0 disables.")

    # Repair to force watertight
    ap.add_argument("--fill_holes", action="store_true",
                    help="Try to fill boundary holes using Open3D tensor mesh fill_holes().")
    ap.add_argument("--hole_size_factor", type=float, default=3.0,
                    help="hole_size = hole_size_factor * alpha_voxel. Increase to close larger gaps, "
                         "but may over-close plant gaps.")

    # Save options
    ap.add_argument("--save_debug_pcd", action="store_true",
                    help="Save alpha-stage pcd after downsample/cleaning/jitter as alpha_pcd.ply")

    return ap.parse_args()


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


# ---------------------------- Utils ----------------------------

def apply_vggt(pcd, invert_z, use_scale, scale):
    pts = np.asarray(pcd.points, dtype=np.float64)
    if invert_z:
        pts[:, 2] *= -1.0
    if use_scale:
        pts *= float(scale)
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd


def remove_bad_points(pcd):
    # remove non-finite first
    pcd.remove_non_finite_points()
    # remove duplicates (important for alpha/tetra)
    pcd.remove_duplicated_points()
    return pcd


def nn_median_distance(pcd, k=2, sample=5000, seed=0):
    pts = np.asarray(pcd.points)
    n = int(len(pts))
    if n < 100:
        raise ValueError("Too few points for NN median estimate.")

    rng = np.random.default_rng(seed)
    if n > sample:
        idx = rng.choice(n, size=sample, replace=False)
    else:
        idx = np.arange(n)

    tree = o3d.geometry.KDTreeFlann(pcd)
    dists = []
    for i in idx:
        _, _, dist2 = tree.search_knn_vector_3d(pcd.points[int(i)], int(k))
        if len(dist2) >= 2:
            dists.append(np.sqrt(dist2[1]))
    if len(dists) == 0:
        raise RuntimeError("NN search failed to produce distances.")
    return float(np.median(np.asarray(dists)))


def normalize_pcd(pcd):
    pts = np.asarray(pcd.points, dtype=np.float64)
    center = pts.mean(axis=0)
    pts0 = pts - center[None, :]
    minv = pts0.min(axis=0)
    maxv = pts0.max(axis=0)
    diag = float(np.linalg.norm(maxv - minv)) + 1e-12
    ptsn = pts0 / diag
    pcdn = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(ptsn))
    return pcdn, center, diag


def denormalize_mesh(mesh, center, diag):
    m = o3d.geometry.TriangleMesh(mesh)
    V = np.asarray(m.vertices, dtype=np.float64)
    V = V * float(diag) + center[None, :]
    m.vertices = o3d.utility.Vector3dVector(V)
    return m


def clean_mesh(mesh):
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    return mesh


def keep_largest_component(mesh):
    try:
        tri_clusters, tri_counts, _ = mesh.cluster_connected_triangles()
        tri_counts = np.asarray(tri_counts)
        if tri_counts.size == 0:
            return mesh
        k = int(tri_counts.argmax())
        mask_remove = tri_clusters != k
        mesh.remove_triangles_by_mask(mask_remove)
        mesh.remove_unreferenced_vertices()
        return mesh
    except Exception:
        return mesh


def mesh_stats(mesh):
    v = int(np.asarray(mesh.vertices).shape[0])
    t = int(np.asarray(mesh.triangles).shape[0])
    return v, t


def fill_holes_tensor(mesh_legacy, hole_size):
    """
    Uses Open3D tensor mesh fill_holes().
    If tensor API is unavailable, returns original mesh.
    """
    try:
        tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh_legacy)
        tmesh2 = tmesh.fill_holes(float(hole_size))
        mesh2 = tmesh2.to_legacy()
        return mesh2
    except Exception:
        return mesh_legacy


def try_alpha_shape(pcd_for_alpha,
                    alpha,
                    do_fill_holes=False,
                    hole_size=None,
                    volume_mult=1.0,
                    volume_unit="unit^3"):
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd_for_alpha, float(alpha))
    mesh = clean_mesh(mesh)
    mesh = keep_largest_component(mesh)
    mesh = clean_mesh(mesh)

    repaired = False
    if do_fill_holes and hole_size is not None and not mesh.is_watertight():
        mesh2 = fill_holes_tensor(mesh, hole_size=float(hole_size))
        mesh2 = clean_mesh(mesh2)
        mesh2 = keep_largest_component(mesh2)
        mesh2 = clean_mesh(mesh2)
        mesh = mesh2
        repaired = True

    wt = bool(mesh.is_watertight())
    mf = bool(mesh.is_edge_manifold())
    v, t = mesh_stats(mesh)

    vol = None
    if wt:
        try:
            # Open3D returns volume in the current length unit cubed.
            # Apply conversion multiplier (e.g., m^3 -> cm^3).
            vol = float(mesh.get_volume()) * float(volume_mult)
        except Exception:
            vol = None

    info = {
        "method": "alpha_shape",
        "alpha": float(alpha),
        "watertight": wt,
        "edge_manifold": mf,
        "V": v,
        "T": t,
        "volume": vol,
        "volume_unit": str(volume_unit),
        "repaired": bool(repaired),
    }
    return mesh, info


# ---------------------------- Main ----------------------------

def main():
    args = parse_args()
    ensure_dir(args.out_dir)

    pcd0 = o3d.io.read_point_cloud(args.in_pcd)
    if pcd0.is_empty():
        raise ValueError("Empty point cloud.")

    # Apply VGGT corrections consistently
    pcd0 = apply_vggt(pcd0, args.invert_z, args.use_scale, args.scale)

    # Volume reporting unit:
    # If scaled to meters (common usage), convert m^3 -> cm^3 by *1e6
    if args.use_scale:
        volume_mult = 1e6
        volume_unit = "cm^3"
    else:
        volume_mult = 1.0
        volume_unit = "unit^3"

    # Build alpha-stage point cloud (THIS is what drives concave hull)
    alpha_voxel = float(args.alpha_voxel)
    if alpha_voxel <= 0:
        raise ValueError("--alpha_voxel must be > 0 for a stable concave hull workflow.")

    pcd_alpha = pcd0.voxel_down_sample(alpha_voxel)

    if args.remove_outlier:
        pcd_alpha, _ = pcd_alpha.remove_statistical_outlier(
            nb_neighbors=int(args.outlier_nb),
            std_ratio=float(args.outlier_std),
        )

    pcd_alpha = remove_bad_points(pcd_alpha)

    # Estimate NN median in alpha space
    nn_med = nn_median_distance(
        pcd_alpha, k=2, sample=int(args.alpha_nn_sample), seed=0
    )

    # Optional jitter to break coplanar/grid degeneracy
    if float(args.jitter_ratio) > 0:
        pts = np.asarray(pcd_alpha.points, dtype=np.float64)
        sigma = float(args.jitter_ratio) * float(nn_med)
        pts = pts + np.random.normal(scale=sigma, size=pts.shape)
        pcd_alpha.points = o3d.utility.Vector3dVector(pts)

    if args.save_debug_pcd:
        o3d.io.write_point_cloud(os.path.join(args.out_dir, "alpha_pcd.ply"), pcd_alpha)

    # Optional normalization for alpha computation
    center = None
    diag = None
    pcd_for_alpha = pcd_alpha
    if args.normalize:
        pcd_for_alpha, center, diag = normalize_pcd(pcd_alpha)
        nn_med = nn_med / float(diag)

    # Alpha sweep
    k_list = [float(x.strip()) for x in args.alpha_k_list.split(",") if x.strip() != ""]
    attempts = []
    selected = None

    hole_size = None
    if args.fill_holes:
        hole_size = float(args.hole_size_factor) * float(alpha_voxel)

    for k in k_list:
        alpha = k * nn_med

        mesh, info = try_alpha_shape(
            pcd_for_alpha,
            alpha=alpha,
            do_fill_holes=bool(args.fill_holes),
            hole_size=hole_size if args.fill_holes else None,
            volume_mult=volume_mult,
            volume_unit=volume_unit,
        )
        info["k"] = float(k)

        # Denormalize back if needed
        if args.normalize and center is not None and diag is not None:
            mesh = denormalize_mesh(mesh, center=center, diag=diag)
            mesh = clean_mesh(mesh)
            mesh = keep_largest_component(mesh)
            mesh = clean_mesh(mesh)

            info["watertight"] = bool(mesh.is_watertight())
            info["edge_manifold"] = bool(mesh.is_edge_manifold())
            info["V"], info["T"] = mesh_stats(mesh)

            if info["watertight"]:
                try:
                    info["volume"] = float(mesh.get_volume()) * float(volume_mult)
                except Exception:
                    info["volume"] = None
            else:
                info["volume"] = None
            info["volume_unit"] = str(volume_unit)

        # Save mesh
        tag = f"k{str(k).replace('.','p')}"
        out_mesh = os.path.join(args.out_dir, f"concave_alpha_{tag}.ply")
        o3d.io.write_triangle_mesh(out_mesh, mesh)
        info["mesh_path"] = out_mesh
        attempts.append(info)

        # Selection criteria
        if info["T"] >= int(args.min_triangles) and info["watertight"] and info["volume"] is not None:
            selected = info
            break

    summary = {
        "input": args.in_pcd,
        "out_dir": args.out_dir,
        "vggt_correction": {
            "invert_z": bool(args.invert_z),
            "use_scale": bool(args.use_scale),
            "scale": float(args.scale),
        },
        "volume_reporting": {
            "volume_unit": str(volume_unit),
            "volume_multiplier_applied_to_open3d_volume": float(volume_mult),
            "assumption_when_use_scale_true": "points are in meters, so Open3D volume is m^3 and is converted to cm^3",
        },
        "alpha_stage": {
            "alpha_voxel": float(alpha_voxel),
            "remove_outlier": bool(args.remove_outlier),
            "outlier_nb": int(args.outlier_nb),
            "outlier_std": float(args.outlier_std),
            "points_alpha": int(len(pcd_alpha.points)),
            "nn_median_alpha": float(nn_med) if not args.normalize else float(nn_med * float(diag)),  # report in original scale
            "normalize_used": bool(args.normalize),
            "jitter_ratio": float(args.jitter_ratio),
        },
        "repair": {
            "fill_holes": bool(args.fill_holes),
            "hole_size_factor": float(args.hole_size_factor),
            "hole_size": float(hole_size) if hole_size is not None else None,
        },
        "alpha_sweep": {
            "k_list": k_list,
            "attempts": attempts,
            "selected": selected,
        }
    }

    out_json = os.path.join(args.out_dir, "summary_concave.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    if selected is not None:
        print("[OK] Found watertight concave hull mesh.")
        print("  k =", selected["k"], "alpha =", selected["alpha"])
        print("  watertight =", selected["watertight"], "edge_manifold =", selected["edge_manifold"])
        print(f"  volume = {selected['volume']} {selected['volume_unit']}")
        print("  mesh =", selected["mesh_path"])
        print("  summary =", out_json)
    else:
        print("[WARN] No watertight concave hull mesh found.")
        print("  Check the saved meshes: concave_alpha_k*.ply")
        print("  Try:")
        print("    1) Increase --hole_size_factor (e.g., 5, 8, 12) with --fill_holes")
        print("    2) Adjust --alpha_k_list (wider range)")
        print("    3) Keep --alpha_voxel around 0.002~0.003 for stability")
        print("    4) Enable --normalize, and/or increase --jitter_ratio slightly (e.g., 0.002)")
        print("  summary =", out_json)


if __name__ == "__main__":
    main()
