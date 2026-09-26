"""
Новый файл, на основе main_3_ftle_fb_volume.ipynb (не менялся).
Считает FTLE вперёд и назад по времени и сохраняет один VTK-файл
(объёмная тетраэдрическая сетка, не облако точек) ровно с четырьмя полями:
FTLE, grad_ftle, FTLE_B, grad_ftle_b.
"""
import os
import re
import time
import numpy as np
import pyvista as pv
from scipy.spatial import KDTree

DATA_DIR = "300"
N_WORKERS = -1  # все ядра CPU

T = 0.8                     # длительность сердечного цикла, с
N = 92                       # число кадров на цикл
dt_frame = T / N
step_rk = dt_frame / 10
LABEL_STEP = 0.01            # шаг метки в имени файла между соседними кадрами
PERIOD_LABEL = N * LABEL_STEP

START_LABEL = None           # None = первый кадр (labels[0])
T_TARGET = 0.15              # окно интегрирования в одну сторону, с
VOXEL_DIVS = 300             # плотность засева начальных частиц
K_VEL = 8                    # соседей для интерполяции скорости
K_GRAD = 20                  # соседей для МНК-Якобиана и градиента
ALPHA_MULT = 3.0             # множитель для alpha-фильтра Delaunay3D

TAG = f"{T_TARGET:g}_{VOXEL_DIVS}"
OUT_VTK = f"aorta_FTLE_FB_volume_{TAG}.vtk"


# ---------- загрузка кадров ----------

def load_vtk(name, verbose=True):
    data = pv.read(os.path.join(DATA_DIR, name))
    if verbose:
        print(data)
    return data


LABEL_PATTERN = re.compile(r"_([0-9]+(?:\.[0-9]+)?)\.vtk$")


def list_frames(data_dir):
    files = [f for f in os.listdir(data_dir) if f.endswith(".vtk")]
    files = [f for f in files if LABEL_PATTERN.search(f)]
    files.sort(key=lambda f: float(LABEL_PATTERN.search(f).group(1)))
    labels = [float(LABEL_PATTERN.search(f).g, labels


# ---------- частицы-затравки ----------

def generate_particles(data):
    particles = data.points.copy()
    print(f"сгенерировано {particles.shape[0]} точек")
    return particles


def voxel_downsample(points, voxel_size):
    idx = np.floor(points / voxel_size).astype(int)
    group = {}
    for i, key in enumerate(map(tuple, idx)):
        if key not in group:
            group[key] = []
        group[key].append(i)
    return [v[0] for v in group.values()]


def remove_particles(points, idx):
    return points[np.asarray(idx)]


# ---------- интерполяция скорости (IDW по k соседям) ----------

def make_field_interp(tree, values, k=8, max_dist=None):
    def interp(X):
        X = np.atleast_2d(X)
        bad = np.isnan(X).any(axis=1)
        Xq = np.where(bad[:, None], 0.0, X)
        dist, idx = tree.query(Xq, k=k, workers=N_WORKERS)
        dist = np.maximum(dist, 1e-12)
        w = 1.0 / dist**2
        w /= w.sum(axis=1, keepdims=True)
        out = np.einsum('nk,nk...->n...', w, values[idx])
        if max_dist is not None:
            bad = bad | out
    return interp


# ---------- Якобиан / Коши-Грин / FTLE (векторизовано) ----------

def compute_all_J_fast(X_0, X_1, tree, k=20):
    dist, idx = tree.query(X_0, k=k, workers=N_WORKERS)
    mask = dist > 0
    A = (X_0[idx] - X_0[:, None, :]) * mask[:, :, None]
    B = (X_1[idx] - X_1[:, None, :]) * mask[:, :, None]
    AtA = np.einsum('nki,nkj->nij', A, A)
    AtB = np.einsum('nki,nkj->nij', A, B)
    x = np.linalg.pinv(AtA) @ AtB
    return np.transpose(x, (0, 2, 1))


def compute_all_CG_fast(J_matrix):
    return np.matmul(np.transpose(J_matrix, (0, 2, 1)), J_matrix)


def compute_all_FTLE_fast(CG_matrix, T_win):
    l_max = np.linalg.eigva(1.0 / (2 * abs(T_win))) * np.log(l_max)


# ---------- RK4 вперёд (direction=+1) и назад (direction=-1) ----------

def vel_func(X, tau, interp_a, interp_b, dt_local, direction=1.0):
    alpha = tau / dt_local
    va = interp_a(X)
    vb = interp_b(X)
    vel = (1 - alpha) * va + alpha * vb
    nan_a = np.isnan(va).any(axis=1)
    nan_b = np.isnan(vb).any(axis=1)
    vel[nan_a & ~nan_b] = vb[nan_a & ~nan_b]
    vel[nan_b & ~nan_a] = va[nan_b & ~nan_a]
    vel[nan_a & nan_b] = np.nan
    return direction * vel


def rk4_step(X, tau, dt, interp_a, interp_b, dt_local, direction=1.0):
    k1 = vel_func(X, tau, interp_a, interp_b, dt_local, direction)
    k2 = vel_func(X + 0.5*dt*k1, tau + 0.5*dt, interp_a, interp_b, dt_local, direction)
    k3 = vel_func(X + 0.5*dt*k2, tau + 0.5*dt, interp_a, interp_b, dt_local, direcal, direction)
    return X + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)


# ---------- кольцевой обход кадров (в 300/ ровно один цикл) ----------

def build_cycle(frames, labels):
    if abs((labels[-1] - labels[0]) - PERIOD_LABEL) < 1e-9:
        return frames[:-1], labels[:-1]  # последний кадр дублирует фазу первого
    return list(frames), list(labels)


def make_gap_frames(cyc_labels):
    n_cyc = len(cyc_labels)

    def gap_frames(i):
        j = (i + 1) % n_cyc
        d = cyc_labels[j] - cyc_labels[i]
        if d <= 0:
            d += PERIOD_LABEL
        return int(round(d / LABEL_STEP))

    return gap_frames, n_cyc


def build_path(start, direction, T_win, gap_frames, n_cyc):
    steps, elapsed, i = [], 0.0, start
    while elapsed < T_win - 1e-12 and len(steps) < n_cyc:
        if direction > 0:
            j = (i + 1) % n_cyc
            dt_local = gap_frames(i) * dt_frame
        else:
            j = (i - 1) % n_cyc
            dt_local = gap_frames(j) * dt_frame
        steps.append((i, j, dt_local))
        elapsed += dt_local
        i = j
    return steps, elapsed


# ---------- интегрирование цепочки кадров ----------

def integrate(X_start, steps, direction, start_U, mesh_tree, max_dist_vel, cyc_frames, tag=""):
    X = X_start.copy()
    cur_interp = make_field_interp(mesh_tree, start_U, k=K_VEL, max_dist=max_dist_vel)

    for s, (i, j, dt_local) in enumerate(steps):
        nxt = load_vtk(cyc_frames[j], verbose=False)
        nxt_interp = make_field_interp(mesh_tree, nxt.point_data["U"], k=K_VEL, max_dist=max_dist_vel)

        n_sub = max(1, round(dt_local / step_rk))
        sub_dt = dt_local / n_sub

        tau = 0.0
        for _ in range(n_sub):
            X = rk4_step(X, tau, sub_dt, cur_interp, nxt_interp, dt_local, direction)
            tau += sub_dt

        alive = (~np.isnan(X).any(axis=1)).sum()
        print(f"[{tag} {s + 1}/{len(steps)}] {cyc_frames[j]}: живых частиц {alive} из {len(X)}")

        del nxt
        cur_interp = nxt_interp

    return X


# ---------- перенос поля с частиц на сетку + градиент (векторизованный МНК) ----------

def field_and_grad_on(src_pts, src_vals, query_pts, k=K_GRAD, min_neigh=4):
    tree = KDTree(src_pts)
    _d, _ = tree.query(src_pts[:2000], k=2, workers=N_WORKERS)
    max_dist = 5 * np.median(_d[:, 1])

    dist, idx = tree.query(query_pts, k=k, workers=N_WORKERS)
    w = 1.0 / np.maximum(dist, 1e-12)**2
    w /= w.sum(axis=1, keepdims=True)
    vals = np.sum(w * src_vals[idx], axis=1)

    mask = dist > 0
    coverage = (dist[:, 0] <= max_dist) & (mask.sum(axis=1) >= min_neigh)

    A = (src_pts[idx] - query_pts[:, None, :]) * mask[:, :, None]
    b = (src_vals[idx] - vals[:, None]) * mask
    AtA = np.einsum('mki,mkj->mij', A, A)
    Atb = np.einsum('mki,mk->mi', A, b)
    grad = (np.linalg.pinv(AtA) @ Atb[..., None])[..., 0]

    vals[~coverage] = np.nan
    grad[~coverage] = np.nan
    return vals, grad, coverage


# ================= основной расчёт =================

if __name__ == "__main__":
    frames, labels = list_frames(DATA_DIR)
    print(f"найдено {len(frames)} кадров, метки от {labels[0]} до {labels[-1]}")

    cyc_frames, cyc_labels = build_cycle(frames, labels)
    gap_frames, n_cyc = make_gap_frames(cyc_labels)

    start_label = START_LABEL if START_LABEL is not None else cyc_labels[0]
    start_i = min(range(n_cyc), key=lambda i: abs(cyc_labels[i] - start_label))

    steps_f, T_f = build_path(start_i, +1, T_TARGET, gap_frames, n_cyc)
    steps_b, T_b = build_path(start_i, -1, T_TARGET, gap_frames, n_cyc)
    print(f"старт: {cyc_frames[start_i]}; вперёд {len(steps_f)} шагов (T_f={T_f:.4f} с), "
          f"назад {len(steps_b)} шагов (T_b={T_b:.4f} с)")

    data_start = load_vtk(cyc_frames[start_i])
    particles = generate_particles(data_start)

    z_length = particles[:, 2].max() - particles[:, 2].min()
    step = z_length / VOXEL_DIVS
    idx = voxel_downsample(particles, step)
    points = remove_particles(particles, idx)
    print(f"всего {len(points)} точек-затравок")

    mesh_tree = KDTree(data_start.points)
    _dd, _ = mesh_tree.query(data_start.points[:5000], k=2, workers=N_WORKERS)
    max_dist_vel = 5 * np.median(_dd[:, 1])
    start_U = data_start.point_data["U"]

    # ---- интегрирование вперёд и назад ----
    t0 = time.perf_counter()
    X_f = integrate(points, steps_f, +1.0, start_U, mesh_tree, max_dist_vel, cyc_frames, tag="вперёд")
    alive_f = ~np.isnan(X_f).any(axis=1)
    X0_f, X1_f = points[alive_f], X_f[alive_f]
    print(f"вперёд: дожило {len(X0_f)} из {len(points)}, {time.perf_counter() - t0:.1f} с")

    t0 = time.perf_counter()
    X_b = integrate(points, steps_b, -1.0, start_U, mesh_tree, max_dist_vel, cyc_frames, tag="назад")
    alive_b = ~np.isnan(X_b).any(axis=1)
    X0_b, X1_b = points[alive_b], X_b[alive_b]
    print(f"назад: дожило {len(X0_b)} из {len(points)}, {time.perf_counter() - t0:.1f} с")

    # ---- FTLE на выживших частицах ----
    tree_f = KDTree(X0_f)
    FTLE_f = compute_all_FTLE_fast(compute_all_CG_fast(compute_all_J_fast(X0_f, X1_f, tree_f, k=K_GRAD)), T_f)
    ok_f = np.isfinite(FTLE_f)
    X0_f_clean, FTLE_f_clean = X0_f[ok_f], FTLE_f[ok_f]

    tree_b = KDTree(X0_b)
    FTLE_b = compute_all_FTLE_fast(compute_all_CG_fast(compute_all_J_fast(X0_b, X1_b, tree_b, k=K_GRAD)), T_b)
    ok_b = np.isfinite(FTLE_b)
    X0_b_clean, FTLE_b_clean = X0_b[ok_b], FTLE_b[ok_b]

    # ---- перенос FTLE/FTLE_B и градиентов на исходную сетку ----
    mesh_pts = data_start.points
    ftle_mesh, grad_ftle, cov_f = field_and_grad_on(X0_f_clean, FTLE_f_clean, mesh_pts)
    ftle_b_mesh, grad_ftle_b, cov_b = field_and_grad_on(X0_b_clean, FTLE_b_clean, mesh_pts)
    print(f"покрытие FTLE: {cov_f.sum()}/{len(cov_f)}, FTLE_B: {cov_b.sum()}/{len(cov_b)}")

    # ---- сборка полей на исходной сетке ----
    mesh = data_start.copy()
    mesh.clear_data()
    mesh.point_data["FTLE"] = ftle_mesh
    mesh.point_data["grad_ftle"] = grad_ftle
    mesh.point_data["FTLE_B"] = ftle_b_mesh
    mesh.point_data["grad_ftle_b"] = grad_ftle_b

    # ---- объёмная (тетраэдрическая) сетка вместо облака точек ----
    # берём только точки, где определены оба поля - без NaN внутри объёма
    cov_both = cov_f & cov_b
    covered = mesh.extract_points(np.where(cov_both)[0], adjacent_cells=False)
    for arr in list(covered.point_data.keys()):
        if arr not in ("FTLE", "grad_ftle", "FTLE_B", "grad_ftle_b"):
            del covered.point_data[arr]

    tree_cov = KDTree(covered.points)
    _dd, _ = tree_cov.query(covered.points[:5000], k=2, workers=N_WORKERS)
    alpha = ALPHA_MULT * np.median(_dd[:, 1])

    volume = covered.delaunay_3d(alpha=alpha)
    for arr in list(volume.cell_data.keys()):
        del volume.cell_data[arr]

    print("поля в выходном файле:", list(volume.point_data.keys()))
    volume.save(OUT_VTK)
    print(f"сохранено: {OUT_VTK} ({volume.n_points} точек, {volume.n_cells} тетраэдров)")