import open3d as o3d
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

# ================== 参数区 ==================

ROOT = "./data_1201/data_M5"
INPUT_PCD = f"{ROOT}/merged_da3.ply"

# VGGT 无绝对尺度，所有阈值都是相对单位
USE_AUTO_SCALE = False      # True: 按场景对角线自动缩放 eps / 平面阈值

# Z 轴 / 平面相关
Z_AXIS_INVERTED = True      # VGGT 的 Z 轴反向：植物在更小 Z -> True

PLANE_DIST_THRESHOLD_MANUAL = 0.005
PLANE_DIST_THRESHOLD_SCALE = 0.005
PLANE_RANSAC_N = 3
PLANE_ITER = 1000

# ========== 关键：关闭 pre-cut（避免不平地面被裁剪缺口） ==========
USE_NONPLANT_HEIGHT_CUT = False   # 原 B 的“非植物高度裁剪”
AUTO_HEIGHT_MARGIN_FACTOR = 4.0   # 保留但默认不用

# 叶片（绿色）HSV 范围
LEAF_H_MIN, LEAF_H_MAX = 30, 95
LEAF_S_MIN, LEAF_V_MIN = 60, 40

# 茎/叶柄（红/黄棕）HSV 范围
STEM_RED_1_MAX = 20
STEM_RED_2_MIN = 160
STEM_YELLOW_MIN, STEM_YELLOW_MAX = 20, 40
STEM_S_MIN, STEM_V_MIN = 40, 40

# DBSCAN 参数
DBSCAN_EPS_MANUAL = 0.012
DBSCAN_MIN_PTS = 120
DBSCAN_EPS_SCALE = 0.02   # USE_AUTO_SCALE=True 时：eps = scene_diag * scale

# ⭐ 保存 cluster 后点云（原始颜色）
SAVE_CLUSTERED_PCD = False
CLUSTERED_OUTPUT_PATH = f"{ROOT}/merged_clustered.ply"

SAVE_EACH_CLUSTER_SEPARATELY = False
EACH_CLUSTER_DIR = f"{ROOT}/clusters_out"

# ⭐ 基于 cluster 点数过滤
CLUSTER_MIN_SIZE = 6000
CLUSTER_MAX_SIZE = None

# ⭐ 聚类可视化点云输出（每个 cluster 一个颜色）
SAVE_VIS_ALL_COLORED = False
VIS_ALL_COLORED_PCD_PATH = f"{ROOT}/merged_vis_all_colored.ply"

SAVE_VIS_CLUSTERS_COLORED = True
VIS_CLUSTERS_COLORED_PCD_PATH = f"{ROOT}/merged_vis_clusters_colored_v4_0.012.ply"

# ⭐ Debug：保存 cluster 点数分布 + 画图
DEBUG_SAVE_CLUSTER_STATS = False
CLUSTER_STATS_PATH = f"{ROOT}/cluster_size_stats.csv"
CLUSTER_PLOT_PATH = f"{ROOT}/cluster_size_stats.png"

# ================== 新增：簇级“地面/侧坡误簇”过滤 ==================
FILTER_GROUNDLIKE_CLUSTERS = True

# ground_ref + k*PLANE_DIST_THRESHOLD：判定“贴地”的中位数阈值
GROUND_MEDIAN_MARGIN_FACTOR = 3.0   # 2~6 调；越小越严格（更容易踢掉贴地簇）

# (p95 - p5) < k*PLANE_DIST_THRESHOLD：判定“很薄”的阈值
GROUND_HEIGHT_RANGE_FACTOR = 4.0    # 3~8 调；越小越严格

# PCA 平面性：eig_small / eig_large < ratio → 很平
GROUND_PLANARITY_RATIO_MAX = 0.02   # 0.01~0.05 调；越大越容易认成地面

# 平面拟合是否只用 non-plant（通常更稳）
USE_NONPLANT_FOR_PLANE_FIT = True

os.makedirs(ROOT, exist_ok=True)
if SAVE_EACH_CLUSTER_SEPARATELY:
    os.makedirs(EACH_CLUSTER_DIR, exist_ok=True)

# =====================================================
# 0. 读点云 & 基本检查
# =====================================================

print("加载点云:", INPUT_PCD)
if not os.path.exists(INPUT_PCD):
    raise FileNotFoundError(f"找不到点云文件: {INPUT_PCD}")

pcd = o3d.io.read_point_cloud(INPUT_PCD)
points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)
N = points.shape[0]
print("点数:", N)

if N < PLANE_RANSAC_N:
    raise SystemExit(f"点云点数太少 (N={N})，无法做 RANSAC 拟合平面。")

# =====================================================
# 1. 全局 HSV：先确定“哪些是植物”
# =====================================================

rgb_255_all = (colors * 255).astype(np.uint8)
hsv_all = cv2.cvtColor(rgb_255_all.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
H, S, V = hsv_all[:, 0], hsv_all[:, 1], hsv_all[:, 2]

leaf_mask_all = (
    (H > LEAF_H_MIN) & (H < LEAF_H_MAX) &
    (S > LEAF_S_MIN) & (V > LEAF_V_MIN)
)

stem_red_all = ((H < STEM_RED_1_MAX) | (H > STEM_RED_2_MIN))
stem_yellow_all = ((H > STEM_YELLOW_MIN) & (H < STEM_YELLOW_MAX))
stem_mask_all = (stem_red_all | stem_yellow_all) & (S > STEM_S_MIN) & (V > STEM_V_MIN)

plant_mask_all = leaf_mask_all | stem_mask_all
nonplant_mask_all = ~plant_mask_all

print("全局植物颜色点数量（叶片+茎）:", int(plant_mask_all.sum()))
print("全局非植物点数量:", int(nonplant_mask_all.sum()))

if plant_mask_all.sum() == 0:
    raise SystemExit("HSV 未检测到任何植物点，请检查颜色阈值。")

# =====================================================
# 2. 拟合地面平面，并计算到平面的 dist
# =====================================================

if USE_AUTO_SCALE:
    bbox = points.max(axis=0) - points.min(axis=0)
    scene_diag = np.linalg.norm(bbox)
    print("场景对角线长度(相对尺度):", scene_diag)

    DBSCAN_EPS = scene_diag * DBSCAN_EPS_SCALE
    PLANE_DIST_THRESHOLD = scene_diag * PLANE_DIST_THRESHOLD_SCALE
else:
    DBSCAN_EPS = DBSCAN_EPS_MANUAL
    PLANE_DIST_THRESHOLD = PLANE_DIST_THRESHOLD_MANUAL

print(f"DBSCAN_EPS = {DBSCAN_EPS:.5f}, PLANE_DIST_THRESHOLD = {PLANE_DIST_THRESHOLD:.5f}")

# --- 平面拟合：优先用 non-plant 点 ---
if USE_NONPLANT_FOR_PLANE_FIT:
    nonplant_idx = np.where(nonplant_mask_all)[0]
    if nonplant_idx.size < PLANE_RANSAC_N:
        print("[WARN] non-plant 点太少，回退到全局点拟合平面。")
        pcd_fit = pcd
        idx_map = None
    else:
        pcd_fit = pcd.select_by_index(nonplant_idx)
        idx_map = nonplant_idx
        print(f"平面拟合使用 non-plant 子点云：{nonplant_idx.size} 点")
else:
    pcd_fit = pcd
    idx_map = None
    print("平面拟合使用全局点云。")

print("RANSAC 拟合主平面（地面/地膜）...")
plane_model, plane_inliers_local = pcd_fit.segment_plane(
    distance_threshold=PLANE_DIST_THRESHOLD,
    ransac_n=PLANE_RANSAC_N,
    num_iterations=PLANE_ITER
)
a, b, c, d = plane_model

# 把 inliers 映射回全局索引
if idx_map is not None:
    plane_inliers = idx_map[np.asarray(plane_inliers_local, dtype=int)]
else:
    plane_inliers = np.asarray(plane_inliers_local, dtype=int)

# 归一化法向，保证 dist 是真实距离
n_norm = float(np.linalg.norm([a, b, c]))
if n_norm < 1e-12:
    raise SystemExit("平面法向范数过小，拟合失败。")
a, b, c, d = a / n_norm, b / n_norm, c / n_norm, d / n_norm

print("平面方程(归一化): %.6fx + %.6fy + %.6fz + %.6f = 0" % (a, b, c, d))
print("平面内点数量:", int(len(plane_inliers)))
print("平面法向 (a,b,c):", (a, b, c))

dist_raw = a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d

if Z_AXIS_INVERTED:
    dist_raw = -dist_raw
    a, b, c, d = -a, -b, -c, -d
    print("根据 Z_AXIS_INVERTED=True：翻转距离与法向，使“植物方向”为正。")

dist = dist_raw

# =====================================================
# 3. 生成“用于聚类的植物候选点”（不做高度裁剪）
# =====================================================

keep_mask = np.ones(N, dtype=bool)

# 如果你后面想开回 non-plant height cut，这里保留逻辑（默认关闭）
if USE_NONPLANT_HEIGHT_CUT:
    inlier_mask_full = np.zeros(N, dtype=bool)
    inlier_mask_full[plane_inliers] = True
    ground_mask_nonplant = inlier_mask_full & nonplant_mask_all
    ground_dists_nonplant = dist[ground_mask_nonplant]

    if ground_dists_nonplant.size > 0:
        ground_top = np.percentile(ground_dists_nonplant, 95)
        margin = AUTO_HEIGHT_MARGIN_FACTOR * PLANE_DIST_THRESHOLD
        height_thresh = ground_top + margin
        cut_mask = nonplant_mask_all & (dist < height_thresh)
        keep_mask[cut_mask] = False
        print(f"non-plant 高度裁剪启用: ground_top={ground_top:.6f}, threshold={height_thresh:.6f}, "
              f"裁掉 non-plant 点数={int(cut_mask.sum())}")
    else:
        print("[WARN] non-plant 平面内点为空，跳过 non-plant 高度裁剪。")

final_plant_mask = plant_mask_all & keep_mask
final_idx = np.where(final_plant_mask)[0]

print(f"最终用于聚类的植物候选点数: {final_idx.size} / {N}")
if final_idx.size == 0:
    raise SystemExit("HSV 后没有剩余植物点，请检查参数。")

pcd_high = pcd.select_by_index(final_idx)

# =====================================================
# 4. 对“植物候选点”做 DBSCAN 聚类
# =====================================================

print("对植物候选点做 DBSCAN 聚类（按株）...")
labels_high = np.array(
    pcd_high.cluster_dbscan(
        eps=DBSCAN_EPS,
        min_points=DBSCAN_MIN_PTS,
        print_progress=True
    )
)

max_label = labels_high.max()
if max_label < 0:
    raise SystemExit("DBSCAN 没聚出任何簇，检查 EPS / MIN_PTS 参数。")

cluster_labels_pos = labels_high[labels_high >= 0]
cluster_sizes = np.bincount(cluster_labels_pos)
print("各簇点数:", cluster_sizes.tolist())

# =====================================================
# 4.1 簇级别过滤：踢掉 ground-like（贴地 + 很薄 + 很平）
# =====================================================

groundlike_mask = np.zeros_like(cluster_sizes, dtype=bool)

if FILTER_GROUNDLIKE_CLUSTERS:
    inlier_mask_full = np.zeros(N, dtype=bool)
    inlier_mask_full[plane_inliers] = True
    ref_dists = dist[inlier_mask_full]

    if ref_dists.size == 0:
        ground_ref = np.percentile(dist, 5)
        print("[WARN] plane inliers 为空，用全局 dist 低分位作为 ground_ref。")
    else:
        # 用 95 分位当作“地面带上缘”参考，避免极少数离群点
        ground_ref = np.percentile(ref_dists, 95)

    ground_median_thresh = ground_ref + GROUND_MEDIAN_MARGIN_FACTOR * PLANE_DIST_THRESHOLD
    ground_range_thresh = GROUND_HEIGHT_RANGE_FACTOR * PLANE_DIST_THRESHOLD

    print(f"[GroundFilter] ground_ref={ground_ref:.6f}")
    print(f"[GroundFilter] median_thresh={ground_median_thresh:.6f}  (k={GROUND_MEDIAN_MARGIN_FACTOR})")
    print(f"[GroundFilter] range_thresh ={ground_range_thresh:.6f}  (k={GROUND_HEIGHT_RANGE_FACTOR})")
    print(f"[GroundFilter] planar_ratio_max={GROUND_PLANARITY_RATIO_MAX}")

    # 逐簇判定
    for cid in range(len(cluster_sizes)):
        local_mask = (labels_high == cid)
        if not np.any(local_mask):
            continue

        global_ids = final_idx[local_mask]
        pts = points[global_ids]
        dist_c = dist[global_ids]

        p5, p95 = np.percentile(dist_c, [5, 95])
        height_range = float(p95 - p5)
        median_h = float(np.median(dist_c))

        # PCA 平面性
        X = pts - pts.mean(axis=0, keepdims=True)
        C = (X.T @ X) / max(len(pts) - 1, 1)
        eigvals = np.linalg.eigvalsh(C)  # 升序
        planar_ratio = float(eigvals[0] / (eigvals[2] + 1e-12))

        is_groundlike = (
            (median_h < ground_median_thresh) and
            (height_range < ground_range_thresh) and
            (planar_ratio < GROUND_PLANARITY_RATIO_MAX)
        )
        groundlike_mask[cid] = is_groundlike

    print("被判定为 ground-like 的簇数量:", int(groundlike_mask.sum()))

# =====================================================
# 5. Debug：cluster 点数统计
# =====================================================

if DEBUG_SAVE_CLUSTER_STATS:
    cluster_ids_all = np.arange(len(cluster_sizes), dtype=int)
    stats_array = np.column_stack((cluster_ids_all, cluster_sizes))

    np.savetxt(
        CLUSTER_STATS_PATH,
        stats_array,
        fmt="%d",
        delimiter=",",
        header="cluster_id,point_count",
        comments=""
    )
    print(f"[DEBUG] 已保存 cluster 点数表到: {CLUSTER_STATS_PATH}")

    plt.figure(figsize=(10, 5))
    plt.bar(cluster_ids_all, cluster_sizes, width=0.8)
    plt.xlabel("Cluster ID")
    plt.ylabel("Point Count")
    plt.title("Distribution of Cluster Sizes")
    plt.tight_layout()
    plt.savefig(CLUSTER_PLOT_PATH, dpi=300)
    print(f"[DEBUG] 已保存 cluster 点数分布图到: {CLUSTER_PLOT_PATH}")
    plt.show()

# =====================================================
# 6. 点数过滤 + ground-like 过滤（组合）
# =====================================================

valid_mask = np.ones_like(cluster_sizes, dtype=bool)

# 先踢掉 ground-like
if FILTER_GROUNDLIKE_CLUSTERS:
    valid_mask &= (~groundlike_mask)

# 点数阈值过滤
if CLUSTER_MIN_SIZE is not None:
    valid_mask &= (cluster_sizes >= CLUSTER_MIN_SIZE)
if CLUSTER_MAX_SIZE is not None:
    valid_mask &= (cluster_sizes <= CLUSTER_MAX_SIZE)

print("各簇是否通过过滤:", valid_mask.tolist())
cluster_ids = np.where(valid_mask)[0]
print("通过过滤后的有效簇数量:", int(cluster_ids.size))

if cluster_ids.size == 0:
    raise SystemExit("所有簇都被过滤掉了，请检查过滤阈值（尤其 ground-like 阈值）。")

# =====================================================
# 7. 把聚类结果映射回“全局索引”
# =====================================================

cluster_assign_global = -1 * np.ones(N, dtype=int)
for local_idx, cid in enumerate(labels_high):
    if cid < 0:
        continue
    if not valid_mask[cid]:
        continue
    global_idx = final_idx[local_idx]
    cluster_assign_global[global_idx] = cid

print("最终保留的 cluster 数量:", int(cluster_ids.size))

# =====================================================
# 8. 保存原始颜色的 cluster 点云
# =====================================================

plant_cluster_mask_global = cluster_assign_global >= 0
cluster_points = points[plant_cluster_mask_global]
cluster_colors = colors[plant_cluster_mask_global]

if SAVE_CLUSTERED_PCD:
    pc_all_clusters = o3d.geometry.PointCloud()
    pc_all_clusters.points = o3d.utility.Vector3dVector(cluster_points)
    pc_all_clusters.colors = o3d.utility.Vector3dVector(cluster_colors)
    o3d.io.write_point_cloud(CLUSTERED_OUTPUT_PATH, pc_all_clusters)
    print(f"已保存所有 cluster 点云到：{CLUSTERED_OUTPUT_PATH}")

if SAVE_EACH_CLUSTER_SEPARATELY:
    unique_clusters = np.unique(cluster_assign_global[plant_cluster_mask_global])
    print("独立保存每个 cluster 到文件夹:", EACH_CLUSTER_DIR)

    for cid in unique_clusters:
        mask = (cluster_assign_global == cid)
        pts = points[mask]
        cols = colors[mask]

        pcd_c = o3d.geometry.PointCloud()
        pcd_c.points = o3d.utility.Vector3dVector(pts)
        pcd_c.colors = o3d.utility.Vector3dVector(cols)

        out_path = os.path.join(EACH_CLUSTER_DIR, f"cluster_{cid}.ply")
        o3d.io.write_point_cloud(out_path, pcd_c)
        print("保存:", out_path)

# =====================================================
# 9. 聚类可视化：灰底 + 彩色 cluster、仅彩色 cluster
# =====================================================

vis_colors_all = np.copy(colors)
vis_colors_all[:] = 0.6

rng = np.random.default_rng(42)
palette = rng.random((cluster_ids.size, 3))

for k_c, cid in enumerate(cluster_ids):
    mask = (cluster_assign_global == cid)
    vis_colors_all[mask] = palette[k_c]

vis_pcd_all = o3d.geometry.PointCloud()
vis_pcd_all.points = o3d.utility.Vector3dVector(points)
vis_pcd_all.colors = o3d.utility.Vector3dVector(vis_colors_all)

vis_cluster_colors = np.zeros_like(cluster_colors)
cluster_ids_for_points = cluster_assign_global[plant_cluster_mask_global]
for k_c, cid in enumerate(cluster_ids):
    mask_local = (cluster_ids_for_points == cid)
    vis_cluster_colors[mask_local] = palette[k_c]

vis_pcd_clusters_colored = o3d.geometry.PointCloud()
vis_pcd_clusters_colored.points = o3d.utility.Vector3dVector(cluster_points)
vis_pcd_clusters_colored.colors = o3d.utility.Vector3dVector(vis_cluster_colors)

print("窗口 1：整体点云（灰底 + 植物簇彩色）")
o3d.visualization.draw_geometries(
    [vis_pcd_all],
    window_name="Full scene (plant clusters colored)",
    width=1280,
    height=720
)

print("窗口 2：仅植物簇（每个 cluster 彩色）")
o3d.visualization.draw_geometries(
    [vis_pcd_clusters_colored],
    window_name="Plant clusters only (cluster-colored)",
    width=1280,
    height=720
)

if SAVE_VIS_ALL_COLORED:
    o3d.io.write_point_cloud(VIS_ALL_COLORED_PCD_PATH, vis_pcd_all)
    print(f"已保存灰底 + 彩色 cluster 点云到：{VIS_ALL_COLORED_PCD_PATH}")

if SAVE_VIS_CLUSTERS_COLORED:
    o3d.io.write_point_cloud(VIS_CLUSTERS_COLORED_PCD_PATH, vis_pcd_clusters_colored)
    print(f"已保存仅 cluster（彩色）点云到：{VIS_CLUSTERS_COLORED_PCD_PATH}")

print("完成。最终保留的 cluster 数量:", int(cluster_ids.size))
