"""
FTLE вперед и назад по времени для аорты.
python main_code/main_the_best_v2.py
"""
import os
import re
import numpy as np
import pyvista as pv
from scipy.spatial import KDTree

DATA_DIR = "300"

T = 0.8               # длительность сердечного цикла, с
N = 92                # число кадров на цикл
dt_frame = T / N
step_rk = dt_frame / 10
LABEL_STEP = 0.01     # шаг меток времени в именах файлов

START_LABEL = None    # None = первый кадр (labels[0])
T_TARGET = 0.4        # окно интегрирования, с
VOXEL_DIV = 300       # плотность засева начальных частиц (шаг = z_length/VOXEL_DIV)
K_VEL = 8             # соседей для интерполяции скорости
K_GRAD = 20           # соседей для Якобиана и переноса FTLE на сетку
ALPHA_MULT = 3.0      # множитель для alpha-фильтра Delaunay3D

OUT_VTK = f"aorta_FTLE_FB_best_{T_TARGET:g}_{VOXEL_DIV}.vtk"


def load_vtk(name):
    return pv.read(os.path.join(DATA_DIR, name))


LABEL_PATTERN = re.compile(r"_([0-9]+(?:\.[0-9]+)?)\.vtk$")


def list_frames(data_dir):
    files = [f for f in os.listdir(data_dir) if f.endswith(".vtk")]
    files = [f for f in files if LABEL_PATTERN.search(f)]
    files.sort(key=lambda f: float(LABEL_PATTERN.search(f).group(1)))
    labels = [float(LABEL_PATTERN.search(f).group(1)) for f in files]
    return files, labels


def step_frames_between(i, labels):
    # сколько шагов dt_frame от кадра i до следующего; за последним кадром цикла идет первый
    j = (i + 1) % len(labels)
    return round((labels[j] - labels[i]) / LABEL_STEP) % N


def generate_particles(data):
    particles = np.asarray(data.points, dtype=float)
    print(f"сгенерировано {particles.shape[0]} точек")
    return particles


def voxel_downsample(points, voxel_size):
    # по одной частице на воксель, частица ставится в его центр
    idx = np.unique(np.floor(points / voxel_size).astype(np.int64), axis=0)
    return (idx + 0.5) * voxel_size


def make_field_interp(tree, values, k=8, max_dist=None):
    def interp(X):
        X = np.atleast_2d(X)
        bad = np.isnan(X).any(axis=1)
        Xq = np.where(bad[:, None], 0.0, X)
        dist, idx = tree.query(Xq, k=k, workers=-1)
        dist = np.maximum(dist, 1e-12)
        w = 1.0 / dist**2
        w /= w.sum(axis=1, keepdims=True)
        out = np.einsum('nk,nk...->n...', w, values[idx])
        if max_dist is not None:
            bad = bad | (dist[:, 0] > max_dist)
        out[bad] = np.nan
        return out
    return interp


def find_k_neighbors(points, tree):
    return tree.query(points, K_GRAD, workers=-1)


def compute_all_J(X_0, X_1, tree):
    # Якобиан каждой частицы методом наименьших квадратов по соседям: B = A J^T
    _, idx = find_k_neighbors(X_0, tree)
    A = X_0[idx] - X_0[:, None, :]
    B = X_1[idx] - X_1[:, None, :]
    AtA = np.einsum('nki,nkj->nij', A, A)
    AtB = np.einsum('nki,nkj->nij', A, B)
    return np.transpose(np.linalg.pinv(AtA) @ AtB, (0, 2, 1))


def compute_all_CG(J_matrix):
    return np.transpose(J_matrix, (0, 2, 1)) @ J_matrix


def compute_all_FTLE(CG_matrix, T_win):
    l_max = np.linalg.eigvalsh(CG_matrix)[:, -1]
    l_max = np.maximum(l_max, 1e-300)
    return np.log(l_max) / (2 * T_win)


def vel_func(X, t, interp, dt_local, direction):
    # interp дает скорость сразу в двух соседних кадрах: столбцы 0-2 и 3-5
    alpha = t / dt_local
    v = interp(X)
    va, vb = v[:, :3], v[:, 3:]
    vel = (1 - alpha) * va + alpha * vb
    return direction * vel


def rk4_step(X, t, dt, interp, dt_local, direction):
    k1 = vel_func(X, t, interp, dt_local, direction)
    k2 = vel_func(X + 0.5*dt*k1, t + 0.5*dt, interp, dt_local, direction)
    k3 = vel_func(X + 0.5*dt*k2, t + 0.5*dt, interp, dt_local, direction)
    k4 = vel_func(X + dt*k3, t + dt, interp, dt_local, direction)
    return X + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)


def integrate(X0, start_i, direction, cur_U, mesh_tree, frames, labels):
    # direction = +1: вперед -1: назад
    X = X0.copy()
    i, n_done, elapsed = start_i, 0, 0.0

    while elapsed < T_TARGET - 1e-12 and n_done < len(frames):
        j = (i + direction) % len(frames)
        next_name = frames[j]
        next_U = load_vtk(next_name).point_data["U"]
        # поля двух кадров склеены, чтобы искать соседей один раз на оба
        interp = make_field_interp(mesh_tree, np.hstack([cur_U, next_U]), k=K_VEL, max_dist=MAX_DIST_VEL)

        n_frames_step = step_frames_between(i if direction > 0 else j, labels)
        dt_local = n_frames_step * dt_frame
        n_sub = max(1, round(dt_local / step_rk))
        sub_dt = dt_local / n_sub

        for m in range(n_sub):
            X = rk4_step(X, m * sub_dt, sub_dt, interp, dt_local, direction)

        elapsed += dt_local
        n_done += 1
        alive_now = ~np.isnan(X).any(axis=1)
        print(f"[{n_done}] {next_name}: t = {elapsed:.4f} с, живых частиц {alive_now.sum()} из {len(X)}")

        i, cur_U = j, next_U

    return X, elapsed


def fd_derivative(f, h, axis):
    # центральная разность, а где соседа нет (NaN) - односторонняя
    fwd = (np.roll(f, -1, axis) - f) / h
    bwd = (f - np.roll(f, 1, axis)) / h
    out = 0.5 * (fwd + bwd)
    out = np.where(np.isnan(bwd), fwd, out)
    out = np.where(np.isnan(fwd), bwd, out)
    return out


def grad_on_voxel_grid(pts, vals, h):
    # значения частиц раскладываются в 3D-массив
    ijk = np.rint(pts / h - 0.5).astype(np.int64)
    k0 = ijk.min(axis=0) - 1
    ijk -= k0
    f = np.full(tuple(ijk.max(axis=0) + 2), np.nan)
    f[tuple(ijk.T)] = vals
    return np.stack([fd_derivative(f, h, a) for a in range(3)], axis=-1), k0


def trilinear_on(grid, k0, h, query_pts):
    # интерполяция по 8 вершинам ячейки решетки; вершины без значения пропускаются
    s = query_pts / h - 0.5 - k0
    i0 = np.floor(s).astype(np.int64)
    t = s - i0
    acc = np.zeros((len(query_pts), 3))
    wsum = np.zeros(len(query_pts))
    for corner in np.ndindex(2, 2, 2):
        idx = np.clip(i0 + corner, 0, np.array(grid.shape[:3]) - 1)
        w = np.prod(np.where(np.array(corner) == 1, t, 1 - t), axis=1)
        v = grid[tuple(idx.T)]
        ok = np.isfinite(v).all(axis=1)
        acc[ok] += w[ok, None] * v[ok]
        wsum[ok] += w[ok]
    out = np.full((len(query_pts), 3), np.nan)
    good = wsum > 0
    out[good] = acc[good] / wsum[good, None]
    return out


def compute_gradient_field(X0_clean, FTLE_clean, mesh_points, step):
    # FTLE с частиц на сетку: среднее по K_GRAD ближайшим частицам с весами 1/d^2
    tree_ftle = KDTree(X0_clean)
    dd_seed, _ = tree_ftle.query(X0_clean[:2000], k=2)
    max_dist_ftle = 5 * np.median(dd_seed[:, 1])

    dist_ftle, idx_ftle = tree_ftle.query(mesh_points, k=K_GRAD, workers=-1)
    w = 1.0 / np.maximum(dist_ftle, 1e-12)**2
    ftle_on_mesh = np.sum(w * FTLE_clean[idx_ftle], axis=1) / w.sum(axis=1)

    coverage = dist_ftle[:, 0] <= max_dist_ftle
    ftle_on_mesh[~coverage] = np.nan

    # градиент считается разностями по решетке частиц и переносится на сетку
    grad_grid, k0 = grad_on_voxel_grid(X0_clean, FTLE_clean, step)
    grad_ftle = trilinear_on(grad_grid, k0, step, mesh_points)
    grad_ftle[~coverage] = np.nan

    return ftle_on_mesh, grad_ftle, coverage


if __name__ == "__main__":
    frames, labels = list_frames(DATA_DIR)
    if round((labels[-1] - labels[0]) / LABEL_STEP) == N:       # последний кадр повторяет фазу первого
        frames, labels = frames[:-1], labels[:-1]
    print(f"найдено {len(frames)} кадров, метки от {labels[0]} до {labels[-1]}")

    start_label = START_LABEL if START_LABEL is not None else labels[0]
    start_i = min(range(len(labels)), key=lambda i: abs(labels[i] - start_label))
    print(f"начало: {frames[start_i]}")

    data_start = load_vtk(frames[start_i])
    particles = generate_particles(data_start)

    z_length = particles[:, 2].max() - particles[:, 2].min()
    step = z_length / VOXEL_DIV
    points = voxel_downsample(particles, step)
    print(f"всего {len(points)} точек-затравок")

    mesh_tree = KDTree(particles)
    _dd, _ = mesh_tree.query(particles[:5000], k=2)
    MAX_DIST_VEL = 5 * np.median(_dd[:, 1])        # дальше от узлов частица считается вышедшей из аорты

    fields = {}
    coverage = np.ones(len(particles), dtype=bool)
    for direction, name, grad_name in ((+1, "FTLE", "grad_ftle"), (-1, "FTLE_B", "grad_ftle_b")):
        # интегрирование: +1 вперед, -1 назад 
        X, T_actual = integrate(points, start_i, direction, data_start.point_data["U"], mesh_tree, frames, labels)

        alive = ~np.isnan(X).any(axis=1)
        X0, X1 = points[alive], X[alive]
        print(f"T_actual = {T_actual:.4f} с, успешно {len(X0)} из {len(points)}")

        #FTLE на выживших частицах
        tree = KDTree(X0)
        J_matrix = compute_all_J(X0, X1, tree)
        CG_matrix = compute_all_CG(J_matrix)
        FTLE_matrix = compute_all_FTLE(CG_matrix, T_actual)

        valid = ~np.isnan(FTLE_matrix)
        X0_clean, FTLE_clean = X0[valid], FTLE_matrix[valid]

        #перенос FTLE и градиента на исходную сетку
        ftle_on_mesh, grad_ftle, covered_now = compute_gradient_field(X0_clean, FTLE_clean, particles, step)
        print(f"покрытие: {covered_now.sum()} из {len(covered_now)}")

        fields[name] = ftle_on_mesh
        fields[grad_name] = grad_ftle
        coverage &= covered_now

    mesh = data_start.copy()
    mesh.clear_data()
    for name, values in fields.items():
        mesh.point_data[name] = values

    #объемная (тетраэдрическая) сетка вместо облака точек
    covered = mesh.extract_points(np.where(coverage)[0], adjacent_cells=False)
    for arr in list(covered.point_data.keys()):
        if arr not in fields:
            del covered.point_data[arr]

    tree_cov = KDTree(covered.points)
    _dd, _ = tree_cov.query(covered.points[:5000], k=2)
    alpha = ALPHA_MULT * np.median(_dd[:, 1])

    volume = covered.delaunay_3d(alpha=alpha)
    for arr in list(volume.cell_data.keys()):
        del volume.cell_data[arr]

    print("поля в выходном файле:", list(volume.point_data.keys()))
    volume.save(OUT_VTK)
    print(f"сохранено: {OUT_VTK} ({volume.n_points} точек, {volume.n_cells} ячеек)")
