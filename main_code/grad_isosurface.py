"""
Поверхности гребней FTLE (ЛКС) по VTK-файлу без стенок.
Запуск из папки с данными:  python main_code/grad_isosurface.py
Результат: <имя входного файла>_ridges_<FIELD>.vtk
"""
import os

import numpy as np
import pyvista as pv
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.spatial import KDTree

INPUT_VTK = "the_best_vtk/aorta_FTLE_FB_best_0.8_300_nowall.vtk"
FIELD = "FTLE"          # "FTLE" - вперед, "FTLE_B" - назад

STEP_MM = 0.768         # шаг решетки, мм (как у затравок в main_the_best)
SIGMA = 1.0             # сглаживание перед производными, в шагах решетки
ANGLE_MAX = 20          # наибольший угол между grad(s) и e, градусы
FTLE_PCT = 70           # FTLE на гребне должен быть выше этого процентиля по объему
SHARP_PCT = 50          # и острота вершины выше этого процентиля
MIN_AREA_MM2 = 20       # куски поверхности меньше этой площади удаляются, мм2

# вершины ячейки решетки в порядке VTK_VOXEL
CORNERS = [(i, j, k) for k in (0, 1) for j in (0, 1) for i in (0, 1)]


def to_grid(mesh, h):
    """FIELD в точках решетки с шагом h: среднее по 8 ближайшим узлам сетки с весами 1/d^2.
    Возвращает значения, маску точек внутри сетки (не дальше h от узла) и координаты всех точек решетки.
    """
    nodes = np.asarray(mesh.points, dtype=float)
    origin = nodes.min(axis=0) - 2 * h
    shape = np.ceil((nodes.max(axis=0) - origin) / h).astype(int) + 3
    X = origin + h * np.indices(shape).reshape(3, -1).T

    tree = KDTree(nodes)
    nearest, _ = tree.query(X, distance_upper_bound=h, workers=-1)
    inside = np.isfinite(nearest)
    dist, idx = tree.query(X[inside], k=8, workers=-1)
    w = 1.0 / np.maximum(dist, 1e-12)**2
    f = np.zeros(len(X))
    f[inside] = np.sum(w * np.asarray(mesh.point_data[FIELD])[idx], axis=1) / w.sum(axis=1)
    return f.reshape(shape), inside.reshape(shape), X


def derivatives(f, inside, h):
    """Градиент и гессиан поля, сглаженного гауссовым фильтром."""
    # снаружи сетки поле продолжается ближайшим значением изнутри, чтобы край не давал ложных производных
    nearest = distance_transform_edt(~inside, return_distances=False, return_indices=True)
    f = f[tuple(nearest)]

    axes = np.eye(3, dtype=int)                 # order фильтра - порядок производной по каждой оси
    grad = np.stack([gaussian_filter(f, SIGMA, order=tuple(axes[a])) for a in range(3)], axis=-1) / h
    hess = np.empty(f.shape + (3, 3))
    for a in range(3):
        for b in range(a, 3):
            hess[..., a, b] = hess[..., b, a] = gaussian_filter(f, SIGMA, order=tuple(axes[a] + axes[b])) / h**2
    return grad, hess


def cross_direction(hess, inside):
    """Направление поперек гребня e и острота вершины (вторая производная вдоль e с обратным знаком)."""
    lam, Q = np.linalg.eigh(hess[inside])       # собственные числа по возрастанию
    e = np.zeros(inside.shape + (3,))
    sharp = np.zeros(inside.shape)
    e[inside] = Q[:, :, 0]
    sharp[inside] = -lam[:, 0]
    return e, sharp


def zero_level(grad, e, inside, X, fields):
    """Поверхность s = (grad, e) = 0 в ячейках решетки, все 8 вершин которых лежат внутри сетки."""
    nx, ny, nz = inside.shape
    full = np.ones((nx - 1, ny - 1, nz - 1), dtype=bool)
    for i, j, k in CORNERS:
        full &= inside[i:nx - 1 + i, j:ny - 1 + j, k:nz - 1 + k]
    cells = np.argwhere(full)
    # номера 8 вершин каждой ячейки, размер (ячейки, 8)
    ids = np.stack([np.ravel_multi_index((cells + c).T, inside.shape) for c in CORNERS], axis=1)
    E = e.reshape(-1, 3)[ids]
    G = grad.reshape(-1, 3)[ids]

    # e определен с точностью до знака, поэтому в каждой ячейке векторы поворачиваются в одну сторону:
    # вдоль их общего направления (главная ось восьми векторов)
    common = np.linalg.eigh(np.einsum('cki,ckj->cij', E, E))[1][:, :, -1]
    sign = np.sign(np.einsum('cki,ci->ck', E, common))
    s = sign * np.einsum('cki,cki->ck', G, E)

    # на гребне s меняет знак вдоль e, то есть grad(s) направлен вдоль e и косинус угла между ними близок к 1
    slope = s @ (2.0 * np.array(CORNERS) - 1)               # grad(s) в ячейке с точностью до множителя
    cos = np.abs(np.einsum('ci,ci->c', slope, common)) / np.linalg.norm(slope, axis=1)

    # у каждой ячейки свои 8 вершин, потому что в общей вершине знак s у соседних ячеек может быть разным
    voxels = pv.UnstructuredGrid({pv.CellType.VOXEL: np.arange(ids.size).reshape(-1, 8)}, X[ids.ravel()])
    voxels.point_data["s"] = s.ravel()
    voxels.cell_data["cos"] = cos
    for name, values in fields.items():
        voxels.point_data[name] = values.ravel()[ids.ravel()]
    return voxels.contour([0.0], scalars="s").clean().cell_data_to_point_data()


def remove_small_pieces(surf):
    """Убирает куски поверхности площадью меньше MIN_AREA_MM2."""
    surf = surf.connectivity().compute_cell_sizes(length=False, volume=False)
    piece = surf.cell_data["RegionId"]
    area = np.bincount(piece, weights=surf.cell_data["Area"])       # площадь каждого куска
    return surf.remove_cells(area[piece] < MIN_AREA_MM2 * 1e-6).clean()


def main():
    mesh = pv.read(INPUT_VTK)
    h = STEP_MM * 1e-3
    f, inside, X = to_grid(mesh, h)
    grad, hess = derivatives(f, inside, h)
    e, sharp = cross_direction(hess, inside)

    surf = zero_level(grad, e, inside, X, {FIELD: f, "sharpness": sharp})

    # пороги - процентили по всему объему
    f_min = np.percentile(f[inside], FTLE_PCT)
    sharp_min = np.percentile(sharp[inside & (sharp > 0)], SHARP_PCT)

    # от поверхности s = 0 остается только то, что похоже на гребень:
    # grad(s) направлен вдоль e, FTLE высокий, вершина острая
    surf = surf.clip_scalar(scalars="cos", value=np.cos(np.radians(ANGLE_MAX)), invert=False)
    surf = surf.clip_scalar(scalars=FIELD, value=f_min, invert=False)
    surf = surf.clip_scalar(scalars="sharpness", value=sharp_min, invert=False)
    ridges = remove_small_pieces(surf)

    for data in (ridges.point_data, ridges.cell_data):          # в файл идут только FTLE и острота
        for name in data.keys():
            if name not in (FIELD, "sharpness"):
                del data[name]

    out = f"{os.path.splitext(INPUT_VTK)[0]}_ridges_{FIELD}.vtk"
    ridges.save(out)
    print(f"решетка {f.shape}, h = {h * 1e3:.3f} мм; пороги: {FIELD} >= {f_min:.2f} 1/с, "
          f"острота >= {sharp_min * 1e-6:.3f} 1/(с мм2)")
    print(f"поверхность гребней: {ridges.area * 1e4:.1f} см2, {ridges.n_cells} ячеек")
    print(f"сохранено: {out}")


if __name__ == "__main__":
    main()