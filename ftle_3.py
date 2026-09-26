import os
import re
import numpy as np
import pyvista as pv
from scipy.spatial import KDTree

DATA_DIR = "300"

T = 0.8              # длительность сердечного цикла, с
N = 92                # число кадров на цикл
dt_frame = T / N
step_rk = dt_frame / 10

START_LABEL = None    # None = первый кадр (labels[0])
T_TARGET = 0.25        # окно интегрирования, с
VOXEL_DIV = 200        # плотность засева начальных частиц (шаг = z_length/VOXEL_DIV)
K_VEL = 8              # соседей для интерполяции скорости
K_GRAD = 20            # соседей для Якобиана и градиента
ALPHA_MULT = 3.0       # множитель для alpha-фильтра Delaunay3D

OUT_VTK = f"aorta_FTLE_grad_{T_TARGET:g}_{VOXEL_DIV}_main3_volume.vtk"


def load_vtk(name):
    data = pv.read(os.path.join(DATA_DIR, name))
    print(data)
    return data


LABEL_PATTERN = re.compile(r"_([0-9]+(?:\.[0-9]+)?)\.vtk$")


def list_frames(data_dir):
    files = [f for f in os.listdir(data_dir) if f.endswith(".vtk")]
    files = [f for f in files if LABEL_PATTERN.search(f)]
    files.sort(key=lambda f: float(LABEL_PATTERN.search(f).group(1)))
    labels = [float(LABEL_PATTERN.search(f).group(1)) for f in files]
    return files, labels


def step_frames_between(i, labels):
    return round((labels[i + 1] - labels[i]) / 0.01)


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
    return np.array([points[i] for i in idx])


def make_field_interp(tree, values, k=8, max_dist=None):
    def interp(X):
        X = np.atleast_2d(X)
        bad = np.isnan(X).any(axis=1)
        Xq = np.where(bad[:, None], 0.0, X)
        dist, idx = tree.query(Xq, k=k)
        dist = np.maximum(dist, 1e-12)
        w = 1.0 / dist**2
        w /= w.sum(axis=1, keepdims=True)
        out = np.einsum('nk,nk...->n...', w, values[idx])
        if max_dist is not None:
            bad = bad | (dist[:, 0] > max_dist)
        out[bad] = np.nan
        return out
    return interp


def find_k_neighbors(point, tree):
    return tree.query(point, 20)


def compute_J(number, X_0, X_1, tree):
    point = X_0[number]
    dist, idx = find_k_neighbors(point, tree)
    neigh = idx[dist > 0]
    A = X_0[neigh] - X_0[number]
    B = X_1[neigh] - X_1[number]
    return np.linalg.lstsq(A, B, rcond=None)[0].T


def compute_all_J(points, X_0, X_1, tree):
    return [compute_J(i, X_0, X_1, tree) for i in range(len(points))]


def compute_CG(J):
    return J.T @ J


def compute_all_CG(points, J_matrix):
    return [compute_CG(J) for J in J_matrix]


def compute_FTLE(CG, T_win):
    l_max = max(np.linalg.eigvalsh(CG))
    l_max = max(l_max, 1e-300)
    return 1 / (2 * T_win) * np.log(l_max)


def compute_all_FTLE(CG_matrix, T_win):
    return [compute_FTLE(m, T_win) for m in CG_matrix]


def vel_func(X, t, interp_a, interp_b, dt_local):
    alpha = t / dt_local
    va = interp_a(X)
    vb = interp_b(X)
    vel = (1 - alpha) * va + alpha * vb
    nan_a = np.isnan(va).any(axis=1)
    nan_b = np.isnan(vb).any(axis=1)
    vel[nan_a & ~nan_b] = vb[nan_a & ~nan_b]
    vel[nan_b & ~nan_a] = va[nan_b & ~nan_a]
    vel[nan_a & nan_b] = np.nan
    return vel


def rk4_step(X, t, dt, interp_a, interp_b, dt_local):
    k1 = vel_func(X, t, interp_a, interp_b, dt_local)
    k2 = vel_func(X + 0.5*dt*k1, t + 0.5*dt, interp_a, interp_b, dt_local)
    k3 = vel_func(X + 0.5*dt*k2, t + 0.5*dt, interp_a, interp_b, dt_local)
    k4 = vel_func(X + dt*k3, t + dt, interp_a, interp_b, dt_local)
    return X + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)


def integrate_forward(X0, start_i, end_i, cur_U, mesh_tree, frames, labels):
    X = X0.copy()
    cur_interp = make_field_interp(mesh_tree, cur_U, k=K_VEL, max_dist=MAX_DIST_VEL)

    for i in range(start_i, end_i):
        next_name = frames[i + 1]
        next_data = load_vtk(next_name)
        next_U = next_data.point_data["U"]
        next_interp = make_field_interp(mesh_tree, next_U, k=K_VEL, max_dist=MAX_DIST_VEL)

        n_frames_step = step_frames_between(i, labels)
        dt_local = n_frames_step * dt_frame
        n_sub = max(1, round(dt_local / step_rk))
        sub_dt = dt_local / n_sub

        t = 0.0
        for _ in range(n_sub):
            X = rk4_step(X, t, sub_dt, cur_interp, next_interp, dt_local)
            t += sub_dt

        alive_now = ~np.isnan(X).any(axis=1)
        print(f"[{i + 1 - start_i}/{end_i - start_i}] {next_name}: живых частиц {alive_now.sum()} из {len(X)}")

        del next_data
        cur_U, cur_interp = next_U, next_interp

    return X


def compute_gradient_field(X0_clean, FTLE_clean, mesh_points):
    tree_ftle = KDTree(X0_clean)
    dd_seed, _ = tree_ftle.query(X0_clean[:2000], k=2)
    max_dist_ftle = 5 * np.median(dd_seed[:, 1])

    dist_ftle, idx_ftle = tree_ftle.query(mesh_points, k=K_GRAD)
    w = 1.0 / np.maximum(dist_ftle, 1e-12)**2
    w /= w.sum(axis=1, keepdims=True)
    ftle_on_mesh = np.sum(w * FTLE_clean[idx_ftle], axis=1)

    coverage = dist_ftle[:, 0] <= max_dist_ftle
    ftle_on_mesh[~coverage] = np.nan

    grad_ftle = np.full((len(mesh_points), 3), np.nan)
    for i in np.where(coverage)[0]:
        p = mesh_points[i]
        neigh = idx_ftle[i][dist_ftle[i] > 0]
        if len(neigh) < 4:
            coverage[i] = False
            ftle_on_mesh[i] = np.nan
            continue
        A = X0_clean[neigh] - p
        b = FTLE_clean[neigh] - ftle_on_mesh[i]
        g, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        grad_ftle[i] = g

    return ftle_on_mesh, grad_ftle, coverage


if __name__ == "__main__":
    frames, labels = list_frames(DATA_DIR)
    print(f"найдено {len(frames)} кадров, метки от {labels[0]} до {labels[-1]}")

    start_label = START_LABEL if START_LABEL is not None else labels[0]
    start_i = min(range(len(labels)), key=lambda i: abs(labels[i] - start_label))

    elapsed, end_i = 0.0, start_i
    while end_i + 1 < len(labels) and elapsed < T_TARGET:
        elapsed += step_frames_between(end_i, labels) * dt_frame
        end_i += 1
    T_actual = elapsed
    print(f"начало: {frames[start_i]}  конец: {frames[end_i]}  кадров: {end_i - start_i}  "
          f"T_actual = {T_actual:.4f} с")

    data_start = load_vtk(frames[start_i])
    particles = generate_particles(data_start)

    z_length = particles[:, 2].max() - particles[:, 2].min()
    step = z_length / VOXEL_DIV
    idx = voxel_downsample(particles, step)
    points = remove_particles(particles, idx)
    print(f"всего {len(points)} точек-затравок")

    mesh_tree = KDTree(data_start.points)
    _dd, _ = mesh_tree.query(data_start.points[:5000], k=2)
    MAX_DIST_VEL = 5 * np.median(_dd[:, 1])

    # ---- интегрирование вперёд ----
    X0 = points.copy()
    X = integrate_forward(X0, start_i, end_i, data_start.point_data["U"], mesh_tree, frames, labels)

    alive = ~np.isnan(X).any(axis=1)
    X0, X1 = X0[alive], X[alive]
    print(f"успешно {len(X0)} из {len(points)}")

    # ---- FTLE на выживших частицах ----
    tree = KDTree(X0)
    J_matrix = compute_all_J(X0, X0, X1, tree)
    CG_matrix = compute_all_CG(X0, J_matrix)
    FTLE_matrix = np.array(compute_all_FTLE(CG_matrix, T_actual))

    valid = ~np.isnan(FTLE_matrix)
    X0_clean, FTLE_clean = X0[valid], FTLE_matrix[valid]

    # ---- перенос FTLE и градиента на исходную сетку ----
    ftle_on_mesh, grad_ftle, coverage = compute_gradient_field(X0_clean, FTLE_clean, data_start.points)
    print(f"покрытие: {coverage.sum()} из {len(coverage)}")

    mesh = data_start.copy()
    mesh.clear_data()
    mesh.point_data["FTLE"] = ftle_on_mesh
    mesh.point_data["gradFTLE"] = grad_ftle

    # ---- объёмная (тетраэдрическая) сетка вместо облака точек ----
    covered = mesh.extract_points(np.where(coverage)[0], adjacent_cells=False)
    for arr in list(covered.point_data.keys()):
        if arr not in ("FTLE", "gradFTLE"):
            del covered.point_data[arr]

    tree_cov = KDTree(covered.points)
    _dd, _ = tree_cov.query(covered.points[:5000], k=2)
    alpha = ALPHA_MULT * np.median(_dd[:, 1])

    volume = covered.delaunay_3d(alpha=alpha)
    for arr in list(volume.cell_data.keys()):
        del volume.cell_data[arr]

    print("поля в выходном файле:", list(volume.point_data.keys()))
    volume.save(OUT_VTK)
    print(f"сохранено: {OUT_VTK} ({volume.n_points} точек, {volume.n_cells} тетраэдров)")