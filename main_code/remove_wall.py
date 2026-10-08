"""
Убирает из VTK-файла слой у границы аорты.
Запуск из папки с данными:  python main_code/remove_wall.py
Результат: <имя входного файла>_nowall.vtk
"""
import os
import warnings

import numpy as np
import pyvista as pv
from scipy.spatial import KDTree

INPUT_VTK = "the_best_vtk/aorta_FTLE_FB_best_0.8_300.vtk"
DEPTH_MM = 2.3          # толщина удаляемого слоя, мм


def boundary_points(mesh):
    """Точки внешней поверхности сетки. Берется самый большой кусок, обрывки от Delaunay3D не считаются."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")         
        surface = mesh.extract_surface().extract_largest()
    return np.asarray(surface.points, dtype=float)


def main():
    mesh = pv.read(INPUT_VTK)

    # расстояние от каждого узла до границы; остается то, что дальше DEPTH_MM
    dist, _ = KDTree(boundary_points(mesh)).query(np.asarray(mesh.points, dtype=float), workers=-1)
    mesh.point_data["wall_dist"] = dist
    inner = mesh.clip_scalar(scalars="wall_dist", value=DEPTH_MM * 1e-3, invert=False)
    del inner.point_data["wall_dist"]

    out = os.path.splitext(INPUT_VTK)[0] + "_nowall.vtk"
    inner.save(out)
    print(f"удален слой {DEPTH_MM:g} мм: объем {mesh.volume * 1e6:.1f} -> {inner.volume * 1e6:.1f} см3, "
          f"точек {mesh.n_points} -> {inner.n_points}")
    print(f"сохранено: {out}")


if __name__ == "__main__":
    main()
