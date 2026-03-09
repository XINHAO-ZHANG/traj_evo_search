#!/usr/bin/env python3
"""
批量生成轨迹可视化图片，用于交互式网页展示
- 输出：static/images/trajectories/{empirical,semantic}/{subtask}_{model}.png
- 支持8个子任务：TSP30, TSP60, oscillator1, oscillator2, OBP-OR3, OBP-Weibull, Summarization, Simplification
- 两种视图：empirical (fitness vs diversity) 和 semantic (MDS embedding)
"""

import os
import re
import math
import ast
import json
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.manifold import MDS

# ============ 配置 ============
OUTPUT_DIR = Path(__file__).parent / "static" / "images" / "trajectories"
EMPIRICAL_DIR = OUTPUT_DIR / "empirical"
SEMANTIC_DIR = OUTPUT_DIR / "semantic"

# 所有子任务的索引文件
INDEX_FILES = {
    # TSP
    "TSP30": "/Users/lievretre/Desktop/evo_eval_sheets/tsp30_raw.csv",
    "TSP60": "/Users/lievretre/Desktop/evo_eval_sheets/tsp60_raw.csv",
    # Symbolic Regression
    "oscillator1": "/Users/lievretre/Desktop/evo_eval_sheets/oscillator1_raw.csv",
    "oscillator2": "/Users/lievretre/Desktop/evo_eval_sheets/oscillator2_raw.csv",
    # Heuristic Design (Binpacking)
    "OBP-OR3": "/Users/lievretre/Desktop/evo_eval_sheets/bin_packing_or3_raw.csv",
    "OBP-Weibull": "/Users/lievretre/Desktop/evo_eval_sheets/bin_packing_weibull_raw.csv",
    # Prompt Optimization
    "Summarization": "/Users/lievretre/Desktop/evo_eval_sheets/promptopt_sum_raw.csv",
    "Simplification": "/Users/lievretre/Desktop/evo_eval_sheets/promptopt_sim_raw.csv",
}

# 任务家族分类
TASK_FAMILIES = {
    "tsp": ["TSP30", "TSP60"],
    "sr": ["oscillator1", "oscillator2"],
    "heuristic": ["OBP-OR3", "OBP-Weibull"],
    "prompt": ["Summarization", "Simplification"],
}

# Prompt Optimization 的预计算 MDS 坐标 parquet 文件
PROMPT_PARQUET_FILES = {
    "Summarization": "/Users/lievretre/Desktop/mds_pack/promptopt_sum_cos_coords.parquet",
    "Simplification": "/Users/lievretre/Desktop/mds_pack/promptopt_sim_cos_coords.parquet",
}

# Heuristic Design (OBP) 的 embedding parquet 根目录
HEURISTIC_PARQUET_ROOT = Path("parquet_embeddings")

# Heuristic 的任务键名映射
HEURISTIC_TASK_KEYS = {
    "OBP-OR3": "binpacking_or3",
    "OBP-Weibull": "binpacking_weibull",
}

# 列名候选（自动探测）
PATH_COL_CANDS = ["csv_file_path", "csv_path", "path"]
GEN_COL_CANDS = ["generation", "gen", "g"]
TYPE_COL_CANDS = ["type", "sample_type", "role"]
FITN_COL_CANDS = ["fitness_normed", "fitness", "score"]
DIVERSITY_RAW_CANDS = ["total_distance", "novelty", "near_distance", "diversity", "distance"]
DIVERSITY_NORMED_FALLBACK = ["total_distance_normed", "novelty_normed", "near_distance_normed"]
MODEL_COL_CANDS = ["model", "alias", "model_alias", "price_key", "engine"]
SEED_COL_CANDS = ["seed", "seed_id", "run"]
TOUR_COL_CANDS = ["genome", "path", "perm", "route", "cycle"]
EMB_COL_CANDS = ["embedding", "behavior", "vector", "embed"]

# MDS 参数 (与原始脚本一致)
MDS_MAX_POINTS = 4000
PER_BUCKET = 60
MDS_KW_TSP = dict(
    n_components=2,
    dissimilarity="precomputed",
    n_init=1,
    max_iter=300,
    eps=1e-3,
    random_state=42,
    verbose=1,
)
MDS_KW_EMB = dict(
    n_components=2,
    dissimilarity="precomputed",
    n_init=1,
    max_iter=300,
    eps=1e-3,
    random_state=42,
    verbose=1,
)
OOS_K = 8
OOS_P = 2.0
RNG = np.random.default_rng(42)

# 视觉样式
mpl.rcParams.update(mpl.rcParamsDefault)
plt.style.use("default")
mpl.rcParams["figure.facecolor"] = "white"
mpl.rcParams["axes.facecolor"] = "white"
mpl.rcParams["savefig.facecolor"] = "white"

SEED_MARKERS = ['o', 's', '^', 'D', 'P', 'X', 'v', '<', '>', '*', 'h', 'H', 'p']


# ============ 工具函数 ============
def seed_to_marker(seed_value) -> str:
    try:
        i = int(seed_value)
    except Exception:
        return 'o'
    return SEED_MARKERS[i % len(SEED_MARKERS)]


def find_col(df: pd.DataFrame, cands: List[str], required: bool = True) -> Optional[str]:
    for c in cands:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"Required column not found. Tried: {cands}")
    return None


def find_first_column(df: pd.DataFrame, cands: List[str]) -> Optional[str]:
    for c in cands:
        if c in df.columns:
            return c
    return None


def try_get_seed_series(df: pd.DataFrame, path: str) -> pd.Series:
    for c in SEED_COL_CANDS:
        if c in df.columns:
            return df[c]
    name = Path(path).name
    m = re.search(r"seed[_\-]?(\d+)", name, flags=re.IGNORECASE)
    if m:
        v = m.group(1)
        return pd.Series([int(v)] * len(df), index=df.index)
    return pd.Series(["unknown"] * len(df), index=df.index)


def load_index(index_path: str) -> pd.DataFrame:
    p = Path(index_path)
    idx = pd.read_csv(p)
    path_col = find_col(idx, PATH_COL_CANDS, required=True)
    keep = [path_col]
    for c in MODEL_COL_CANDS + SEED_COL_CANDS:
        if c in idx.columns:
            keep.append(c)
    out = idx[keep].copy()
    out.rename(columns={path_col: "csv_file_path"}, inplace=True)
    return out


def read_process_csv(process_csv_path: str) -> Optional[pd.DataFrame]:
    p = Path(process_csv_path)
    if not p.exists():
        print(f"[warn] missing process CSV: {p}")
        return None
    try:
        return pd.read_csv(p)
    except Exception as e:
        print(f"[warn] read failed: {p} -> {e}")
        return None


def pareto_front_xy(x: np.ndarray, y: np.ndarray, direction: str = "rtl") -> np.ndarray:
    ok = np.isfinite(x) & np.isfinite(y)
    if not ok.any():
        return np.empty((0, 2))
    xv = x[ok]
    yv = y[ok]
    if direction == "rtl":
        order = np.argsort(xv)[::-1]
        xv = xv[order]
        yv = yv[order]
        best = -np.inf
        pts = []
        for xi, yi in zip(xv, yv):
            if yi > best:
                pts.append((xi, yi))
                best = yi
        return np.array(pts) if pts else np.empty((0, 2))
    else:
        order = np.argsort(xv)
        xv = xv[order]
        yv = yv[order]
        best = -np.inf
        pts = []
        for xi, yi in zip(xv, yv):
            if yi > best:
                pts.append((xi, yi))
                best = yi
        return np.array(pts) if pts else np.empty((0, 2))


def get_lo_hi(vec, exclude_01=True, robust_q=(1, 99)):
    vec = vec.astype(float)
    if exclude_01:
        vec = vec[(vec != 0.0) & (vec != 1.0)]
    if len(vec) == 0:
        return 0.0, 1.0
    if robust_q:
        lo, hi = np.nanpercentile(vec, robust_q)
    else:
        lo, hi = np.nanmin(vec), np.nanmax(vec)
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        lo, hi = 0.0, 1.0
    return lo, hi


def sanitize_model_name(name: str) -> str:
    """清理模型名用于文件名"""
    # 移除非法字符
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', str(name))
    # 移除或替换其他问题字符
    sanitized = sanitized.replace(' ', '_').replace(',', '_')
    return sanitized[:100]  # 限制长度


def robust_minmax(x: np.ndarray, q=(1, 99)) -> np.ndarray:
    """鲁棒的 min-max 归一化"""
    lo, hi = np.nanpercentile(x, q)
    den = (hi - lo) if hi > lo else 1.0
    return np.clip((x - lo) / den, 0, 1)


def robust_minmax_reverse(x: np.ndarray, q=(0, 100)) -> np.ndarray:
    """鲁棒的反向 min-max 归一化（用于 Prompt Optimization）"""
    x_np = np.asanyarray(x)
    lo, hi = np.nanpercentile(x_np, q)
    den = (hi - lo) if hi > lo else 1.0
    return np.clip((hi - x_np) / den, 0, 1)


# ============ TSP 特定函数 ============
def coerce_tour(x):
    """把 tour 解析成 list[int]"""
    if isinstance(x, list):
        return [int(v) for v in x]
    if isinstance(x, np.ndarray):
        return [int(v) for v in x.tolist()]
    if isinstance(x, str):
        try:
            v = ast.literal_eval(x)
            return [int(t) for t in (v.tolist() if isinstance(v, np.ndarray) else list(v))]
        except Exception:
            toks = re.findall(r"\d+", x)
            return [int(t) for t in toks]
    raise ValueError(f"Unsupported tour type: {type(x)}")


def edge_indexer(n_cities: int):
    idx = {}
    col = 0
    for i in range(n_cities):
        for j in range(i + 1, n_cities):
            idx[(i, j)] = col
            col += 1
    return idx, col


def tours_to_incidence(tours, n_cities: int, edge_map):
    S = len(tours)
    E = len(edge_map)
    A = np.zeros((S, E), dtype=np.uint8)
    for r, t in enumerate(tours):
        m = len(t)
        for k in range(m):
            a = int(t[k])
            b = int(t[(k + 1) % m])
            if a == b:
                continue
            if a > b:
                a, b = b, a
            if (a, b) in edge_map:
                A[r, edge_map[(a, b)]] = 1
    return A


def tsp_distance_matrix(incidence_matrix: np.ndarray, n_edges: int) -> np.ndarray:
    """TSP 边差距离: D = 1 - |E∩|/n_edges"""
    inter = (incidence_matrix @ incidence_matrix.T).astype(np.float32)
    D = 1.0 - inter / float(n_edges)
    np.fill_diagonal(D, 0.0)
    return D


# ============ Embedding 距离函数 ============
def parse_embedding(x):
    """解析 embedding"""
    if isinstance(x, (list, np.ndarray)):
        return np.array(x, dtype=np.float32)
    if isinstance(x, str):
        try:
            v = ast.literal_eval(x)
            return np.array(v, dtype=np.float32)
        except Exception:
            return None
    return None


def cosine_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    """余弦距离矩阵"""
    # 归一化
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    normalized = embeddings / norms
    # 余弦相似度
    sim = normalized @ normalized.T
    # 距离
    D = 1.0 - sim
    np.fill_diagonal(D, 0.0)
    return D.astype(np.float32)


# ============ 采样和 OOS 插值 ============
def stratified_sample_indices(df: pd.DataFrame, per_bucket=PER_BUCKET, total_cap=MDS_MAX_POINTS) -> np.ndarray:
    df_reset = df.reset_index(drop=True)
    parts = []
    for (m, g), sub in df_reset.groupby(["model", "generation"], sort=False):
        if len(sub) > per_bucket:
            parts.append(sub.sample(per_bucket, random_state=int(RNG.integers(1 << 31))))
        else:
            parts.append(sub)
    out = pd.concat(parts, ignore_index=False)
    if len(out) > total_cap:
        out = out.sample(total_cap, random_state=42)
    return out.index.to_numpy()


def oos_place_by_blocks(D_func, fit_idx, Y_fit, N, k=8, p=2.0, block=4000):
    """OOS 插值放置"""
    m = len(fit_idx)
    Y = np.zeros((N, 2), dtype=np.float32)
    Y[fit_idx] = Y_fit

    not_mask = np.ones(N, dtype=bool)
    not_mask[fit_idx] = False
    q_idx = np.where(not_mask)[0]
    if len(q_idx) == 0:
        return Y

    for s in range(0, len(q_idx), block):
        sl = q_idx[s: s + block]
        Dq = D_func(sl, fit_idx)  # (len(sl), m)

        if k < m:
            nn = np.argpartition(Dq, kth=k, axis=1)[:, :k]
            rows = np.arange(len(sl))[:, None]
            dnn = Dq[rows, nn]
        else:
            nn = np.tile(np.arange(m), (len(sl), 1))
            dnn = Dq

        w = 1.0 / (dnn + 1e-8) ** p
        w = w / w.sum(axis=1, keepdims=True)
        Y[sl] = (w[..., None] * Y_fit[nn]).sum(axis=1).astype(np.float32)

    return Y


# ============ 绘图函数 ============
def plot_empirical_single(df: pd.DataFrame, subtask: str, model: str, 
                          gnorm_global: Normalize, cmap, output_path: Path):
    """绘制单个 empirical 图（fitness vs diversity）"""
    fig, ax = plt.subplots(figsize=(4.5, 4))
    
    colors = cmap(gnorm_global(df["generation"].to_numpy()))
    seeds = df["seed"].astype(str).fillna("unknown")
    uniq_seeds = seeds.unique().tolist()

    for s in uniq_seeds:
        mask = seeds.eq(s).to_numpy()
        mk = seed_to_marker(s)
        ax.scatter(df.loc[mask, "fitness_normed"], df.loc[mask, "diversity_gnormed"],
                   s=18, marker=mk, c=colors[mask], edgecolor="white", linewidth=0.35, alpha=0.95, zorder=2)

    # Pareto frontier
    pts = pareto_front_xy(
        df["fitness_normed"].to_numpy(float),
        df["diversity_gnormed"].to_numpy(float),
        direction="rtl"
    )
    if pts.size:
        ax.plot(pts[:, 0], pts[:, 1], '-', color="steelblue", lw=1.5, alpha=0.85, zorder=1.5)

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, color="#eee", linewidth=0.8, zorder=0)
    ax.set_xlabel("Normalized Fitness")
    ax.set_ylabel("Normalized Diversity")
    ax.set_title(f"{subtask} | {model}", fontsize=10)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=gnorm_global)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label("Generation")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_semantic_single(df: pd.DataFrame, subtask: str, model: str,
                         gnorm_global: Normalize, cmap, output_path: Path,
                         is_tsp: bool = False):
    """绘制单个 semantic MDS 图"""
    fig, ax = plt.subplots(figsize=(4.5, 4))
    
    if "mds_x" not in df.columns or "mds_y" not in df.columns:
        ax.text(0.5, 0.5, "No MDS data", ha='center', va='center', fontsize=12)
        ax.set_title(f"{subtask} | {model}", fontsize=10)
        plt.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return

    dfp = df.sort_values("generation").copy()
    colors = cmap(gnorm_global(dfp["generation"].to_numpy()))
    
    # 点大小按 fitness（TSP 和 SR 都使用 10-90 范围）
    fit_vals = dfp["fitness_normed"].to_numpy()
    sizes = 10 + 80 * fit_vals  # 10-90 范围，与参考脚本一致

    ax.scatter(dfp["mds_x"], dfp["mds_y"], s=sizes, c=colors,
               edgecolor="white", linewidth=0.35, alpha=0.92, zorder=2)

    ax.set_title(f"{subtask} | {model}", fontsize=10, pad=4)
    
    # 统一样式：无坐标轴刻度、无网格（与参考脚本一致）
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=gnorm_global)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label("Generation")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ============ 数据加载函数 ============
def load_empirical_data(subtask: str, index_csv: str) -> Dict[str, pd.DataFrame]:
    """加载 empirical 数据（fitness vs diversity）"""
    panels = {}
    idx = load_index(index_csv)
    
    for _, row in idx.iterrows():
        path = str(row["csv_file_path"]).strip()
        df = read_process_csv(path)
        if df is None or df.empty:
            continue

        gen_col = find_col(df, GEN_COL_CANDS, required=True)
        fit_col = find_col(df, FITN_COL_CANDS, required=True)
        
        div_raw_col = find_first_column(df, DIVERSITY_RAW_CANDS)
        if div_raw_col is None:
            div_raw_col = find_first_column(df, DIVERSITY_NORMED_FALLBACK)
            if div_raw_col is None:
                continue

        model_col = find_col(df, MODEL_COL_CANDS, required=False)
        seed_series = try_get_seed_series(df, path)

        use_cols = [gen_col, fit_col, div_raw_col]
        if model_col:
            use_cols.append(model_col)
        sub = df[use_cols].copy()
        sub.rename(columns={
            gen_col: "generation",
            fit_col: "fitness_normed",
            div_raw_col: "diversity_raw",
        }, inplace=True)
        sub["seed"] = seed_series.values
        sub["subtask"] = subtask

        # 获取模型名
        if model_col and model_col in df.columns and df[model_col].notna().any():
            model_name = str(df[model_col].dropna().iloc[0])
        else:
            model_name = None
            for c in MODEL_COL_CANDS:
                if c in row and isinstance(row[c], str) and row[c].strip():
                    model_name = row[c]
                    break
            if not model_name:
                model_name = Path(path).stem

        if model_name not in panels:
            panels[model_name] = pd.DataFrame()
        panels[model_name] = pd.concat([panels[model_name], sub], ignore_index=True)

    # 清理和归一化
    cleaned_panels = {}
    for model, df in panels.items():
        if df.empty:
            continue
        df["generation"] = pd.to_numeric(df["generation"], errors="coerce")
        df["fitness_normed"] = pd.to_numeric(df["fitness_normed"], errors="coerce")
        df["diversity_raw"] = pd.to_numeric(df["diversity_raw"], errors="coerce")
        df = df.dropna(subset=["generation", "fitness_normed", "diversity_raw"])
        df = df.sort_values("generation").reset_index(drop=True)
        cleaned_panels[model] = df

    # 按 subtask 归一化
    if cleaned_panels:
        all_div = pd.concat([df["diversity_raw"] for df in cleaned_panels.values()], ignore_index=True)
        lo, hi = get_lo_hi(all_div)
        den = hi - lo if hi > lo else 1.0
        for model, df in cleaned_panels.items():
            df["diversity_gnormed"] = ((df["diversity_raw"] - lo) / den).clip(0.0, 1.0)

    return cleaned_panels


def load_tsp_semantic_data(subtask: str, index_csv: str) -> Dict[str, pd.DataFrame]:
    """加载 TSP 的 semantic MDS 数据（与原始脚本一致）"""
    idx = load_index(index_csv)
    
    rows = []
    for _, row in idx.iterrows():
        csv_path = str(row["csv_file_path"]).strip()
        df = read_process_csv(csv_path)
        if df is None or df.empty:
            continue

        gen_col = find_col(df, GEN_COL_CANDS, required=True)
        fit_col = find_col(df, FITN_COL_CANDS, required=True)
        tour_col = find_col(df, TOUR_COL_CANDS, required=True)
        model_col = find_col(df, MODEL_COL_CANDS, required=False)

        if model_col and model_col in df.columns and df[model_col].notna().any():
            model_name = str(df[model_col].dropna().iloc[0])
        else:
            model_name = None
            for c in MODEL_COL_CANDS:
                if c in row and isinstance(row[c], str) and row[c].strip():
                    model_name = row[c]
                    break
            if not model_name:
                model_name = Path(csv_path).stem

        sub = pd.DataFrame({
            "generation": pd.to_numeric(df[gen_col], errors="coerce"),
            "fitness": pd.to_numeric(df[fit_col], errors="coerce"),  # 原始 fitness，后续用 robust_minmax 归一化
            "tour_raw": df[tour_col],
            "model": model_name,
        })
        sub = sub.dropna(subset=["generation", "fitness", "tour_raw"])
        sub["tour"] = sub["tour_raw"].apply(coerce_tour)
        rows.append(sub)

    if not rows:
        return {}

    all_data = pd.concat(rows, ignore_index=True)
    all_data = all_data.reset_index(drop=True)
    
    # 计算 MDS
    tours = all_data["tour"].tolist()
    n_cities = len(tours[0])  # 使用 tour 长度作为城市数
    edge_map, E = edge_indexer(n_cities)  # E = n_cities*(n_cities-1)/2
    A = tours_to_incidence(tours, n_cities, edge_map)
    n_edges = n_cities  # 每条环的边数 = 城市数
    
    all_data["model"] = all_data["model"].astype(str)
    N = len(all_data)
    
    print(f"[mds] TSP {subtask}: {N} points | n_cities={n_cities} | n_edges={n_edges}")
    
    # 抽样学 MDS
    if N <= MDS_MAX_POINTS:
        fit_idx = np.arange(N)
    else:
        fit_idx = stratified_sample_indices(all_data)
    
    A_fit = A[fit_idx]
    # 距离矩阵: D = 1 - |E∩| / n_edges
    D_fit = tsp_distance_matrix(A_fit, n_edges)
    
    print(f"[mds] fitting on {len(fit_idx)} points (of {N})")
    mds = MDS(**MDS_KW_TSP)
    Y_fit = mds.fit_transform(D_fit)
    
    # OOS 插值（如果需要）
    if N > MDS_MAX_POINTS:
        def D_func(q_idx, f_idx):
            Aq = A[q_idx]
            Af = A[f_idx]
            inter = (Aq @ Af.T).astype(np.float32)
            return 1.0 - inter / float(n_edges)
        
        Y = oos_place_by_blocks(D_func, fit_idx, Y_fit, N, k=OOS_K, p=OOS_P, block=6000)
    else:
        Y = Y_fit
    
    all_data["mds_x"] = Y[:, 0]
    all_data["mds_y"] = Y[:, 1]
    
    # 使用 robust_minmax 归一化 fitness（与原始脚本一致）
    all_data["fitness_normed"] = robust_minmax(all_data["fitness"].to_numpy(), q=(1, 99))
    
    # 按模型拆分
    panels = {}
    for model, group in all_data.groupby("model"):
        panels[model] = group[["generation", "fitness_normed", "mds_x", "mds_y", "model"]].copy()
        panels[model] = panels[model].reset_index(drop=True)
    
    return panels


def l2_normalize_rows(A: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2 归一化每一行"""
    n = np.linalg.norm(A, axis=1, keepdims=True)
    n = np.where(n > eps, n, 1.0)
    return A / n


def cosine_D_fit(A_fit: np.ndarray) -> np.ndarray:
    """
    A_fit: (m, P) float32，已按行 L2 归一化
    返回 D_fit: (m, m) float32，D = 1 - (A_fit @ A_fit^T)
    """
    S = np.clip(A_fit @ A_fit.T, -1.0, 1.0).astype(np.float32)
    D = (1.0 - S).astype(np.float32)
    np.fill_diagonal(D, 0.0)
    return D


def cosine_D_query_to_fit(Aq: np.ndarray, Af: np.ndarray, block: int = 4000) -> np.ndarray:
    """
    Aq: (b, P)  Af: (m, P)  都需已 L2 归一化
    返回 Dq: (b, m) float32
    """
    b = Aq.shape[0]
    m = Af.shape[0]
    out = np.zeros((b, m), dtype=np.float32)
    for s in range(0, b, block):
        sl = slice(s, min(b, s + block))
        S = np.clip(Aq[sl] @ Af.T, -1.0, 1.0).astype(np.float32)
        out[sl] = 1.0 - S
    return out


def oos_place_by_knn_cos(A_all, fit_idx, Y_fit, k=8, p=2.0, block=4000):
    """
    OOS 插值：基于余弦距离的 KNN-Shepard
    A_all: (N, P) 全体向量
    fit_idx: (m,) 基点索引
    Y_fit: (m, 2) MDS 结果
    返回 Y: (N, 2)
    """
    N = A_all.shape[0]
    Y = np.zeros((N, 2), dtype=np.float32)
    Y[fit_idx] = Y_fit

    mask = np.ones(N, dtype=bool)
    mask[fit_idx] = False
    q_idx = np.where(mask)[0]
    if len(q_idx) == 0:
        return Y

    Af = l2_normalize_rows(A_all[fit_idx].astype(np.float32, copy=False))
    for s in range(0, len(q_idx), block):
        ids = q_idx[s: s + block]
        Aq = l2_normalize_rows(A_all[ids].astype(np.float32, copy=False))
        Dq = cosine_D_query_to_fit(Aq, Af, block=8000)  # (b, m)

        # 取每行的 k 近邻
        m = Af.shape[0]
        if k < m:
            nn = np.argpartition(Dq, kth=k, axis=1)[:, :k]
            rows = np.arange(len(ids))[:, None]
            dnn = Dq[rows, nn]
        else:
            nn = np.tile(np.arange(m), (len(ids), 1))
            dnn = Dq

        w = 1.0 / (dnn + 1e-8) ** p
        w = w / w.sum(axis=1, keepdims=True)
        Y[ids] = (w[..., None] * Y_fit[nn]).sum(axis=1).astype(np.float32)

    return Y


def load_embedding_semantic_data(subtask: str, index_csv: str) -> Dict[str, pd.DataFrame]:
    """加载基于 embedding 的 semantic MDS 数据（SR）- 与参考脚本一致"""
    idx = load_index(index_csv)
    
    rows = []
    for _, row in idx.iterrows():
        csv_path = str(row["csv_file_path"]).strip()
        df = read_process_csv(csv_path)
        if df is None or df.empty:
            continue

        gen_col = find_col(df, GEN_COL_CANDS, required=True)
        fit_col = find_col(df, FITN_COL_CANDS, required=True)
        emb_col = find_first_column(df, EMB_COL_CANDS)
        
        if emb_col is None:
            continue

        model_col = find_col(df, MODEL_COL_CANDS, required=False)

        if model_col and model_col in df.columns and df[model_col].notna().any():
            model_name = str(df[model_col].dropna().iloc[0])
        else:
            model_name = None
            for c in MODEL_COL_CANDS:
                if c in row and isinstance(row[c], str) and row[c].strip():
                    model_name = row[c]
                    break
            if not model_name:
                model_name = Path(csv_path).stem

        sub = pd.DataFrame({
            "generation": pd.to_numeric(df[gen_col], errors="coerce"),
            "fitness_raw": pd.to_numeric(df[fit_col], errors="coerce"),  # 原始 fitness
            "embedding_raw": df[emb_col],
            "model": model_name,
        })
        sub = sub.dropna(subset=["generation", "fitness_raw", "embedding_raw"])
        
        # 解析 embedding
        sub["emb"] = sub["embedding_raw"].apply(parse_embedding)
        sub = sub.dropna(subset=["emb"]).reset_index(drop=True)
        if not sub.empty:
            rows.append(sub[["generation", "fitness_raw", "emb", "model"]])

    if not rows:
        return {}

    all_data = pd.concat(rows, ignore_index=True)
    all_data = all_data.reset_index(drop=True)
    
    # 过滤 embedding 长度不一致的
    lens = all_data["emb"].apply(lambda a: len(a) if a is not None else 0)
    if lens.nunique() > 1:
        mode_len = int(lens.mode().iloc[0])
        all_data = all_data[lens.eq(mode_len)].copy().reset_index(drop=True)
    
    if all_data.empty:
        return {}
    
    # 构建 embedding 矩阵
    A = np.stack(all_data["emb"].tolist()).astype(np.float32)  # (N, P)
    all_data["model"] = all_data["model"].astype(str)
    N, P = A.shape
    
    print(f"[mds] SR {subtask}: {N} points | embedding dim={P}")
    
    # 抽样学 MDS
    if N <= MDS_MAX_POINTS:
        fit_idx = np.arange(N)
    else:
        fit_idx = stratified_sample_indices(all_data)
    
    Af = l2_normalize_rows(A[fit_idx])  # (m, P)
    D_fit = cosine_D_fit(Af)
    
    print(f"[mds] fitting on {len(fit_idx)} points (of {N})")
    mds = MDS(**MDS_KW_EMB)
    Y_fit = mds.fit_transform(D_fit)
    
    # OOS 插值（如果需要）
    if N > MDS_MAX_POINTS:
        Y = oos_place_by_knn_cos(A, fit_idx, Y_fit, k=OOS_K, p=OOS_P, block=4000)
    else:
        Y = Y_fit
    
    all_data["mds_x"] = Y[:, 0]
    all_data["mds_y"] = Y[:, 1]
    
    # 使用 robust_minmax 归一化 fitness（与参考脚本一致）
    all_data["fitness_normed"] = robust_minmax(all_data["fitness_raw"].to_numpy(), q=(1, 99))
    
    # 按模型拆分
    panels = {}
    for model, group in all_data.groupby("model"):
        panels[model] = group[["generation", "fitness_normed", "mds_x", "mds_y", "model"]].copy()
        panels[model] = panels[model].reset_index(drop=True)
    
    return panels


def load_prompt_semantic_data(subtask: str) -> Dict[str, pd.DataFrame]:
    """加载 Prompt Optimization 的 semantic MDS 数据（从预计算的 parquet 文件）"""
    parquet_path = PROMPT_PARQUET_FILES.get(subtask)
    if not parquet_path or not Path(parquet_path).exists():
        print(f"[warn] Missing parquet file for {subtask}: {parquet_path}")
        return {}
    
    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        print(f"[error] Failed to read parquet {parquet_path}: {e}")
        return {}
    
    # 检查必要的列
    required_cols = ["generation", "fitness", "mds_x", "mds_y", "model"]
    for col in required_cols:
        if col not in df.columns:
            print(f"[error] Missing required column '{col}' in {parquet_path}")
            return {}
    
    df["subtask"] = subtask
    
    # 使用反向归一化（与参考脚本一致）
    df["fitness_normed"] = robust_minmax_reverse(df["fitness"].to_numpy(), q=(0, 100))
    
    # 模型名映射：从 parquet 中的长名称映射到 empirical index 中的名称
    MODEL_NAME_MAP = {
        "deepseek_deepseek-chat-v3-0324:free_np2_nc5_simple": "deepseek-v3-chat",
        "google_gemma-3n-e4b-it_np2_nc5_simple": "gemma-3n-4b",
        "meta-llama_llama-3.1-70b-instruct_np2_nc5_simple": "llama-3.1-70b-instruct",
        "meta-llama_llama-3.1-8b-instruct_np2_nc5_simple": "llama-3.1-8b-instruct",
        "meta-llama_llama-3.2-1b-instruct_np2_nc5_simple": "meta-llama-3.2-1b-instruct",
        "meta-llama_llama-3.2-3b-instruct_np2_nc5_simple": "llama-3.2-3b-instruct",
        "mistralai_magistral-small-2506_np2_nc5_simple": "mistral-magistral-small",
        "mistralai_mistral-7b-instruct-v0.3_np2_nc5_simple": "mistral-7b-instruct",
        "mistralai_mistral-small-3.2-24b-instruct:free_np2_nc5_simple": "mistral-24b-instruct",
        "openai_gpt-3.5-turbo_np2_nc5_simple": "gpt-3.5-turbo",
        "openai_gpt-4o-mini_np2_nc5_simple": "gpt-4o-mini",
        "openai_gpt-4o_np2_nc5_simple": "gpt-4o",
        "vertex_ai_gemini-1.5-flash_np2_nc5_simple": "gemini-1.5-flash",
        "vertex_ai_gemini-1.5_np2_nc5_simple": "gemini-1.5-pro",
        "vertex_ai_mistral-large_np2_nc5_simple": "mistral-large",
    }
    
    def map_model_name(name):
        name = str(name)
        if name in MODEL_NAME_MAP:
            return MODEL_NAME_MAP[name]
        # 如果没有精确匹配，尝试清理
        name = name.replace("_np2_nc5_simple", "").replace(":free", "")
        last_underscore = name.rfind("_")
        if last_underscore >= 0:
            return name[last_underscore + 1:]
        return name
    
    df["model_clean"] = df["model"].apply(map_model_name)
    
    print(f"[prompt] {subtask}: {len(df)} points | models: {df['model_clean'].nunique()}")
    
    # 按清理后的模型名分组
    panels = {}
    for model_clean, group in df.groupby("model_clean"):
        panels[model_clean] = group[["generation", "fitness_normed", "mds_x", "mds_y", "model_clean"]].copy()
        panels[model_clean] = panels[model_clean].rename(columns={"model_clean": "model"})
        panels[model_clean] = panels[model_clean].reset_index(drop=True)
    
    return panels


def _collect_parquet_candidates(csv_path: str, task_key: str) -> List[Path]:
    """收集可能的 parquet 文件路径"""
    import glob
    p = Path(csv_path)
    cands = []
    # 1) 同目录同名
    local = p.with_suffix(".parquet")
    if local.exists():
        cands.append(local)
    # 2) 根目录下按分区存的
    root = HEURISTIC_PARQUET_ROOT
    if root.exists():
        # a) write_to_dataset 分区：task_key=…/model=…/*.parquet
        patt1 = str(root / f"task_key={task_key}" / "model=*/**/*.parquet")
        cands += [Path(x) for x in glob.glob(patt1, recursive=True)]
        patt1b = str(root / f"task_key={task_key}" / "**/*.parquet")
        cands += [Path(x) for x in glob.glob(patt1b, recursive=True)]
        # b) 平铺
        patt2 = str(root / "**/*.parquet")
        cands += [Path(x) for x in glob.glob(patt2, recursive=True)]
    # 去重
    uniq = []
    seen = set()
    for q in cands:
        if q not in seen and q.exists():
            uniq.append(q)
            seen.add(q)
    return uniq


def _read_parquet_df(pq_path: Path) -> Optional[pd.DataFrame]:
    """读取 parquet 文件"""
    try:
        return pd.read_parquet(pq_path)
    except Exception:
        try:
            return pd.read_parquet(pq_path, engine="fastparquet")
        except Exception as e:
            print(f"[warn] failed reading parquet: {pq_path} -> {e}")
            return None


def _attach_embeddings_from_parquet(df_csv: pd.DataFrame,
                                    csv_path: str,
                                    task_key: str) -> Optional[pd.DataFrame]:
    """
    从外部 parquet 文件附加 embedding 到 CSV 数据
    返回包含 generation / fitness_raw / type / model / emb 的 DataFrame
    """
    gen_col = find_col(df_csv, GEN_COL_CANDS, required=True)
    fit_col = find_col(df_csv, FITN_COL_CANDS, required=True)
    type_col = find_col(df_csv, TYPE_COL_CANDS, required=False)

    parquets = _collect_parquet_candidates(csv_path, task_key)
    if not parquets:
        print(f"[warn] no parquet found for: {csv_path}")
        return None

    for pq_path in parquets:
        pdf = _read_parquet_df(pq_path)
        if pdf is None or pdf.empty:
            continue

        # 统一 row_idx 命名
        col_rowidx = None
        if "row_idx" in pdf.columns:
            col_rowidx = "row_idx"
        elif "row_id" in pdf.columns:
            col_rowidx = "row_id"

        # 兼容 fitness/raw 命名
        if "fitness_raw" in pdf.columns:
            pdf["_fitness_from_pq"] = pd.to_numeric(pdf["fitness_raw"], errors="coerce")
        elif "fitness" in pdf.columns:
            pdf["_fitness_from_pq"] = pd.to_numeric(pdf["fitness"], errors="coerce")
        else:
            pdf["_fitness_from_pq"] = np.nan

        # 尝试通过 row_idx 直连
        if col_rowidx is not None:
            tmp_csv = df_csv.reset_index(drop=False).rename(columns={"index": "row_id"})
            merged = tmp_csv.merge(pdf, on="row_id", how="left", suffixes=("", "_pq"))
            if "embedding" in merged.columns and merged["embedding"].notna().any():
                out = pd.DataFrame({
                    "generation": pd.to_numeric(merged[gen_col], errors="coerce"),
                    "fitness_raw": pd.to_numeric(merged[fit_col], errors="coerce"),
                    "type": (merged[type_col].astype(str) if type_col in merged.columns else "unknown"),
                    "model": None,
                    "emb": merged["embedding"].apply(
                        lambda v: np.asarray(v, dtype=np.float32) if isinstance(v, (list, np.ndarray)) else np.nan
                    ),
                })
                out = out.dropna(subset=["generation", "fitness_raw", "emb"])
                return out

        # 若没有 row_idx，尝试同名文件过滤
        stem = Path(csv_path).stem
        if "source_csv" in pdf.columns:
            subpdf = pdf[pdf["source_csv"].astype(str).str.contains(stem, na=False)].copy()
        elif "file_source" in pdf.columns:
            subpdf = pdf[pdf["file_source"].astype(str).str.contains(stem, na=False)].copy()
        else:
            subpdf = pdf.copy()
        
        if not subpdf.empty and "embedding" in subpdf.columns:
            # 按顺序对齐
            n = min(len(df_csv), len(subpdf))
            if n > 0:
                out = pd.DataFrame({
                    "generation": pd.to_numeric(df_csv[gen_col].iloc[:n], errors="coerce"),
                    "fitness_raw": pd.to_numeric(df_csv[fit_col].iloc[:n], errors="coerce"),
                    "type": (df_csv[type_col].iloc[:n].astype(str) if type_col in df_csv.columns else "unknown"),
                    "model": None,
                    "emb": [np.asarray(x, dtype=np.float32) if isinstance(x, (list, np.ndarray)) else np.nan
                            for x in subpdf["embedding"].iloc[:n].tolist()],
                })
                out = out.dropna(subset=["generation", "fitness_raw", "emb"])
                return out

    print(f"[warn] failed to attach embeddings from parquet for: {csv_path}")
    return None


def load_heuristic_semantic_data(subtask: str, index_csv: str) -> Dict[str, pd.DataFrame]:
    """加载 Heuristic Design (OBP) 的 semantic MDS 数据（从外部 parquet 读取 embedding）"""
    task_key = HEURISTIC_TASK_KEYS.get(subtask, subtask.lower().replace("-", "_"))
    idx = load_index(index_csv)
    
    rows = []
    for _, irow in idx.iterrows():
        csv_path = str(irow["csv_file_path"]).strip()
        df = read_process_csv(csv_path)
        if df is None or df.empty:
            continue

        # 从外部 parquet 获取 embedding
        sub = _attach_embeddings_from_parquet(df, csv_path, task_key=task_key)
        if sub is None or sub.empty:
            continue

        # 获取模型名
        model_col = find_col(df, MODEL_COL_CANDS, required=False)
        if model_col and model_col in df.columns and df[model_col].notna().any():
            model_name = str(df[model_col].dropna().iloc[0])
        else:
            model_name = None
            for c in MODEL_COL_CANDS:
                if c in irow and isinstance(irow[c], str) and irow[c].strip():
                    model_name = irow[c]
                    break
            if not model_name:
                model_name = Path(csv_path).stem

        sub["model"] = model_name
        rows.append(sub[["generation", "fitness_raw", "emb", "model"]])

    if not rows:
        print(f"[warn] no rows collected for {subtask}")
        return {}

    all_data = pd.concat(rows, ignore_index=True)
    all_data = all_data.reset_index(drop=True)
    
    # 过滤 embedding 长度不一致的
    lens = all_data["emb"].apply(lambda a: len(a) if isinstance(a, np.ndarray) else 0)
    if lens.eq(0).all():
        print(f"[warn] empty embeddings for {subtask}")
        return {}
    mode_len = int(lens[lens > 0].mode().iloc[0])
    all_data = all_data[lens.eq(mode_len)].copy().reset_index(drop=True)
    
    if all_data.empty:
        print(f"[warn] embeddings empty after length filtering for {subtask}")
        return {}
    
    # 构建 embedding 矩阵
    A = np.stack(all_data["emb"].tolist()).astype(np.float32)  # (N, P)
    all_data["model"] = all_data["model"].astype(str)
    N, P = A.shape
    
    print(f"[mds] Heuristic {subtask}: {N} points | embedding dim={P}")
    
    # 抽样学 MDS
    if N <= MDS_MAX_POINTS:
        fit_idx = np.arange(N)
    else:
        fit_idx = stratified_sample_indices(all_data)
    
    Af = l2_normalize_rows(A[fit_idx])  # (m, P)
    D_fit = cosine_D_fit(Af)
    
    print(f"[mds] fitting on {len(fit_idx)} points (of {N})")
    mds = MDS(**MDS_KW_EMB)
    Y_fit = mds.fit_transform(D_fit)
    
    # OOS 插值
    if N > MDS_MAX_POINTS:
        Y = oos_place_by_knn_cos(A, fit_idx, Y_fit, k=OOS_K, p=OOS_P, block=4000)
    else:
        Y = Y_fit
    
    all_data["mds_x"] = Y[:, 0]
    all_data["mds_y"] = Y[:, 1]
    
    # 使用 robust_minmax 归一化 fitness
    all_data["fitness_normed"] = robust_minmax(all_data["fitness_raw"].to_numpy(), q=(1, 99))
    
    # 按模型拆分
    panels = {}
    for model, group in all_data.groupby("model"):
        panels[model] = group[["generation", "fitness_normed", "mds_x", "mds_y", "model"]].copy()
        panels[model] = panels[model].reset_index(drop=True)
    
    return panels


# ============ 主生成函数 ============
def generate_all_images():
    """生成所有轨迹图片"""
    EMPIRICAL_DIR.mkdir(parents=True, exist_ok=True)
    SEMANTIC_DIR.mkdir(parents=True, exist_ok=True)
    
    metadata = {
        "subtasks": list(INDEX_FILES.keys()),
        "task_families": TASK_FAMILIES,
        "models": {},
    }
    
    cmap = plt.get_cmap("viridis")
    
    for subtask, index_csv in INDEX_FILES.items():
        print(f"\n{'='*50}")
        print(f"Processing subtask: {subtask}")
        print(f"{'='*50}")
        
        if not Path(index_csv).exists():
            print(f"[SKIP] Index file not found: {index_csv}")
            continue
        
        # 1. 生成 Empirical 图
        print(f"\n[Empirical] Loading data...")
        empirical_panels = load_empirical_data(subtask, index_csv)
        
        if empirical_panels:
            g_all = pd.concat([df["generation"] for df in empirical_panels.values()], ignore_index=True)
            gnorm_global = Normalize(vmin=g_all.min(), vmax=g_all.max())
            
            models_list = []
            for model, df in empirical_panels.items():
                safe_model = sanitize_model_name(model)
                output_path = EMPIRICAL_DIR / f"{subtask}_{safe_model}.png"
                print(f"  Generating: {output_path.name}")
                plot_empirical_single(df, subtask, model, gnorm_global, cmap, output_path)
                models_list.append({"name": model, "safe_name": safe_model})
            
            metadata["models"][subtask] = models_list
        else:
            print(f"[WARN] No empirical data for {subtask}")
        
        # 2. 生成 Semantic 图
        print(f"\n[Semantic] Loading data...")
        
        is_tsp = subtask in ["TSP30", "TSP60"]
        is_sr = subtask in ["oscillator1", "oscillator2"]
        is_prompt = subtask in ["Summarization", "Simplification"]
        is_heuristic = subtask in ["OBP-OR3", "OBP-Weibull"]
        
        if is_tsp:
            semantic_panels = load_tsp_semantic_data(subtask, index_csv)
        elif is_sr:
            semantic_panels = load_embedding_semantic_data(subtask, index_csv)
        elif is_prompt:
            semantic_panels = load_prompt_semantic_data(subtask)
        elif is_heuristic:
            semantic_panels = load_heuristic_semantic_data(subtask, index_csv)
        else:
            semantic_panels = {}
        
        if semantic_panels:
            g_all = pd.concat([df["generation"] for df in semantic_panels.values()], ignore_index=True)
            gnorm_global = Normalize(vmin=g_all.min(), vmax=g_all.max())
            
            for model, df in semantic_panels.items():
                safe_model = sanitize_model_name(model)
                output_path = SEMANTIC_DIR / f"{subtask}_{safe_model}.png"
                print(f"  Generating: {output_path.name}")
                plot_semantic_single(df, subtask, model, gnorm_global, cmap, output_path, is_tsp=is_tsp)
        else:
            print(f"[WARN] No semantic data for {subtask}")
    
    # 保存 metadata
    metadata_path = OUTPUT_DIR / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n✅ Metadata saved to: {metadata_path}")
    
    print(f"\n{'='*50}")
    print("✅ All images generated!")
    print(f"   Empirical: {EMPIRICAL_DIR}")
    print(f"   Semantic: {SEMANTIC_DIR}")
    print(f"{'='*50}")


if __name__ == "__main__":
    generate_all_images()
