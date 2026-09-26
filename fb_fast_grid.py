"""
fb_fast_grid.py - FTLE вперёд и назад на СТРУКТУРИРОВАННОЙ (регулярной) сетке.

Основан на fb_fast.py (он не менялся). Отличия:
  * затравки - центры занятых вокселей (а не первый узел CFD в вокселе),
    т.е. узлы регулярной решётки с шагом h = Lz / VOXEL_DIVS;
  * у каждой затравки хранится индекс (i, j, k), поэтому соседи известны;
  * градиент деформации F = dX_T/dX_0 и grad(FTLE) считаются конечными
    разностями по решётке (центральные внутри, односторонние у стенки),
    а не МНК по k ближайшим соседям;
  * результат - pv.ImageData (.vti): FTLE, grad_ftle, FTLE_B, grad_ftle_b,
    inside (1 - узел внутри аорты). Вне аорты значения NaN.

Интегрирование (RK4, линейная интерполяция по времени между кадрами,
IDW-интерполяция скорости по K_VEL узлам CFD) - как в fb_fast.py.
"""
import os
import re
import time
import numpy as np
import pyvista as pv
from scipy.spatial import KDTree
from scipy import ndimage

DATA_DIR = "300"
N_WORKERS = -1  # все ядра CPU

T = 0.8                     # длительность сердечного цикла, с
N = 92                      # число кадров на цикл
dt_frame = T / N
step_rk = dt_frame / 10
LABEL_STEP = 0.01           # шаг метки в имени файла между соседними кадрами
PERIOD_LABEL = N * LABEL_STEP

START_LABEL = None          # None = первый кадр (labels[0])
T_TARGET = 0.15             # окно интегрирования в одну сторону, с
VOXEL_DIVS = 300            # шаг решётки h = Lz / VOXEL_DIVS
K_VEL = 8                   # соседей для интерполяции скорости
FILL_HOLES = True           # заполнять пустые воксели внутри аорты

TAG = f"{T_TARGET:g}_{VOXEL_DIVS}"
OUT_VTI = f"aorta_FTLE_FB_grid_{TAG}.vti"
OUT_NPZ = f"aorta_FTLE_FB_grid_{TAG}.npz"   # сырые данные (позиции частиц и т.п.)


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
    labels = [float(LABEL_PATTERN.search(f).group(1)) for f in files]
    return files, labels


# ---------- структурированная сетка затравок (центры вокселей) ----------

def build_voxel_grid(points, h, fill_holes=True):
    """
    Разбивает пространство на воксели с шагом h (как voxel_downsample в fb_fast),
    но в качестве затравки берёт ЦЕНТР каждого занятого вокселя.

    Возвращает:
      mask   - bool-массив (nx, ny, nz): узел решётки внутри аорты;
      origin - координаты узла (0, 0, 0);
      ijk    - (M, 3) индексы узлов-затравок;
      seeds  - (M, 3) координаты затравок = origin + ijk * h.
    Решётка дополнена 1 слоем пустых узлов по краям.
    """
    keys = np.floor(points / h).astype(np.int64)
    k0 = keys.min(axis=0) - 1
    dims = keys.max(axis=0) - k0 + 2
    mask = np.zeros(dims, dtype=bool)
    mask[tuple((keys - k0).T)] = True
    n_occ = int(mask.sum())

    if fill_holes:
        # одиночные пустые воксели внутри потока (узел CFD не попал в воксель)
        nb = sum(np.roll(mask, s, axis=a) for a in range(3) for s in (1, -1))
        mask |= (nb >= 5)
        # полностью замкнутые полости
        mask = ndimage.binary_fill_holes(mask)
    print(f"решётка {tuple(dims)}, h = {h:.4g}; занято вокселей {n_occ}, "
          f"после заполнения дыр {int(mask.sum())}")

    origin = (k0 + 0.5) * h
    ijk = np.argwhere(mask)
    seeds = origin + ijk * h
    return mask, origin, ijk, seeds


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
            bad = bad | (dist[:, 0] > max_dist)
        out[bad] = np.nan
        return out
    return interp


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
    k3 = vel_func(X + 0.5*dt*k2, tau + 0.5*dt, interp_a, interp_b, dt_local, direction)
    k4 = vel_func(X + dt*k3, tau + dt, interp_a, interp_b, dt_local, direction)
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


# ---------- конечные разности на решётке ----------

def fd_derivative(f, h, axis):
    """
    Производная поля f (nx, ny, nz, ...) по оси axis.
    Центральная разность, если оба соседа определены (не NaN);
    односторонняя, если определён только один; иначе NaN.
    """
    pad = [(0, 0)] * f.ndim
    pad[axis] = (1, 1)
    g = np.pad(f, pad, constant_values=np.nan)
    n = f.shape[axis]
    c = np.take(g, np.arange(1, n + 1), axis=axis)
    p = np.take(g, np.arange(2, n + 2), axis=axis)
    m = np.take(g, np.arange(0, n), axis=axis)

    fwd = (p - c) / h
    bwd = (c - m) / h
    out = 0.5 * (fwd + bwd)                     # = (p - m) / 2h
    only_f = np.isnan(bwd) & ~np.isnan(fwd)
    only_b = np.isnan(fwd) & ~np.isnan(bwd)
    out[only_f] = fwd[only_f]
    out[only_b] = bwd[only_b]
    return out


def to_grid(values, ijk, shape):
    """Раскладывает значения в узлах-затравках в массив решётки (NaN вне)."""
    tail = values.shape[1:]
    g = np.full(tuple(shape) + tail, np.nan)
    g[tuple(ijk.T)] = values
    return g


def ftle_on_grid(X_T, ijk, shape, h, T_win):
    """
    X_T - конечные позиции частиц (M, 3), NaN для вылетевших.
    F[..., c, a] = d x_c / d X_a  (как J в fb_fast), C = F^T F.
    """
    Xg = to_grid(X_T, ijk, shape)                        # (nx, ny, nz, 3)
    F = np.stack([fd_derivative(Xg, h, a) for a in range(3)], axis=-1)
    ok = np.isfinite(F).all(axis=(-2, -1)) & np.isfinite(Xg).all(axis=-1)

    ftle = np.full(tuple(shape), np.nan)
    Fv = F[ok]
    C = np.matmul(np.transpose(Fv, (0, 2, 1)), Fv)
    l_max = np.maximum(np.linalg.eigvalsh(C)[:, -1], 1e-300)
    ftle[ok] = np.log(l_max) / (2.0 * abs(T_win))
    return ftle


def grad_on_grid(field, h):
    return np.stack([fd_derivative(field, h, a) for a in range(3)], axis=-1)


# ---------- сохранение ImageData ----------

def vtk_order(a):
    """(nx, ny, nz[, 3]) -> плоский массив в порядке точек vtkImageData (x быстрее всех)."""
    if a.ndim == 3:
        return np.ascontiguousarray(a.transpose(2, 1, 0)).ravel()
    return np.ascontiguousarray(a.transpose(2, 1, 0, 3)).reshape(-1, a.shape[3])


def make_image(shape, h, origin):
    return pv.ImageData(dimensions=tuple(int(n) for n in shape),
                        spacing=(h, h, h), origin=tuple(float(o) for o in origin))


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
    mesh_pts = np.asarray(data_start.points, dtype=float)

    # ---- структурированная сетка затравок ----
    z_length = np.ptp(mesh_pts[:, 2])
    h = z_length / VOXEL_DIVS
    mask, origin, ijk, points = build_voxel_grid(mesh_pts, h, fill_holes=FILL_HOLES)
    shape = mask.shape
    print(f"всего {len(points)} точек-затравок (центры вокселей)")

    mesh_tree = KDTree(mesh_pts)
    _dd, _ = mesh_tree.query(mesh_pts[:5000], k=2, workers=N_WORKERS)
    max_dist_vel = 5 * np.median(_dd[:, 1])
    start_U = data_start.point_data["U"]

    # ---- интегрирование вперёд и назад ----
    t0 = time.perf_counter()
    X_f = integrate(points, steps_f, +1.0, start_U, mesh_tree, max_dist_vel, cyc_frames, tag="вперёд")
    print(f"вперёд: дожило {(~np.isnan(X_f).any(axis=1)).sum()} из {len(points)}, "
          f"{time.perf_counter() - t0:.1f} с")

    t0 = time.perf_counter()
    X_b = integrate(points, steps_b, -1.0, start_U, mesh_tree, max_dist_vel, cyc_frames, tag="назад")
    print(f"назад: дожило {(~np.isnan(X_b).any(axis=1)).sum()} из {len(points)}, "
          f"{time.perf_counter() - t0:.1f} с")

    # ---- FTLE и градиенты конечными разностями ----
    ftle_f = ftle_on_grid(X_f, ijk, shape, h, T_f)
    ftle_b = ftle_on_grid(X_b, ijk, shape, h, T_b)
    grad_f = grad_on_grid(ftle_f, h)
    grad_b = grad_on_grid(ftle_b, h)
    print(f"FTLE определён в {np.isfinite(ftle_f).sum()} узлах, "
          f"FTLE_B - в {np.isfinite(ftle_b).sum()} из {int(mask.sum())}")

    # ---- сохранение ----
    grid = make_image(shape, h, origin)
    grid.point_data["FTLE"] = vtk_order(ftle_f)
    grid.point_data["grad_ftle"] = vtk_order(grad_f)
    grid.point_data["FTLE_B"] = vtk_order(ftle_b)
    grid.point_data["grad_ftle_b"] = vtk_order(grad_b)
    grid.point_data["inside"] = vtk_order(mask.astype(np.uint8))
    grid.save(OUT_VTI)
    print(f"сохранено: {OUT_VTI} (решётка {shape}, шаг {h:.4g})")

    np.savez_compressed(OUT_NPZ, h=h, origin=origin, shape=np.array(shape), ijk=ijk,
                        X0=points, X_f=X_f, X_b=X_b, T_f=T_f, T_b=T_b)
    print(f"сохранено: {OUT_NPZ}")
