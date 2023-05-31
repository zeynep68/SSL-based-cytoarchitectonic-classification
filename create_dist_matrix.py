import multiprocessing
import os
import time

import h5py
import numpy as np
import potpourri3d as pp3d
import pygeodesic.geodesic as geodesic
import siibra
from memory_profiler import profile


class Colors:
    """ https://stackoverflow.com/questions/287871/how-do-i-print-colored-text-to-the-terminal"""
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    ENDC = '\033[0m'


def get_space(name='colin27', format='neuroglancer/precompmesh/surface'):
    space = siibra.spaces[name].get_template().fetch(format=format)
    return space['verts'], space['faces']


def compute_geodesic_distance(idx):
    if MODE == 'pygeodesic':
        return solver.geodesicDistances(np.array([idx]))[0].astype(np.float16)
    elif MODE == 'potpourri3d':
        return solver.compute_distance(idx).astype(np.float16)


def init_solver(vertices, faces):
    if MODE == 'pygeodesic':  # runs way too slow
        """ https://pypi.org/project/pygeodesic/ """
        return geodesic.PyGeodesicAlgorithmExact(vertices, faces)
    elif MODE == 'potpourri3d':
        """ https://github.com/nmwsharp/potpourri3d """
        return pp3d.MeshHeatMethodDistanceSolver(vertices, faces)


@profile
def write_dist_matrix_to_hdf5(h5_file, end_idx=342_042, steps=5_000):  # end_idx = 342_042
    start = time.time()
    for offset in range(0, end_idx, steps):
        dist_rows = compute_multiple_dist_matrix_rows(offset, steps)
        write_rows_to_dist_matrix_in_hdf5(h5_file, np.array(dist_rows), offset)

        if (end_idx - offset) < steps:  # last iteration
            steps = end_idx - offset

    diff = time.time() - start
    # for readability
    print(Colors.OKBLUE + '-' * 60)
    print(f'{diff} secs = {diff / 60} mins')
    print('-' * 60 + Colors.ENDC)


def compute_multiple_dist_matrix_rows(start_idx, num_coordinates):
    """ For each coordinate compute the distance to all the other points
        and save it as a row in the numpy array. """
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        iter_steps = list(range(start_idx, start_idx + num_coordinates))  # results are in correct order

        results = pool.map(compute_geodesic_distance, iter_steps)
    return np.array(results)


def hdf5_file_exists(path):
    if os.path.isfile(path):

        print(Colors.WARNING + 'Warning: File already exists.' + Colors.ENDC)

        choice = input('Do you want to continue (C) or stop (S) the code? ')

        if choice != 'C':
            exit()

    return h5py.File(path, 'a')  # append additional group


def write_rows_to_dist_matrix_in_hdf5(h5_file, rows, start_idx):
    """
    Create a distance matrix in the hdf5 file.
    Row X corresponds to the data point index X in the coordinates array.
    Row X gives the distances from data point X to all other data points.
    Columns correspond to the data point indices according to the coordinates array.
    """
    if start_idx == 0:
        h5_file.create_dataset('dist_matrix', data=rows, chunks=True,
                               maxshape=(None, rows.shape[1]))
    else:
        dataset = h5_file['dist_matrix']
        dataset.resize((dataset.shape[0] + len(rows), dataset.shape[1]))
        dataset[-len(rows):] = rows


def main(dist_matrix=False, coordinates=True, kd_tree=False):
    global solver  # to make it work with multiprocessing
    global MODE  # to make it work with multiprocessing

    MODE = 'potpourri3d'  # 'potpourri3d' or 'pygeodesic'
    path = './geodesic_distance.h5'  # store hdf5 file here

    h5_file = hdf5_file_exists(path=path)

    if coordinates:
        vertices, _ = get_space()
        h5_file.create_dataset('coordinates', data=vertices)  # float64

    if dist_matrix:
        vertices, faces = get_space()
        solver = init_solver(vertices=vertices, faces=faces)
        write_dist_matrix_to_hdf5(h5_file=h5_file, end_idx=342_042, steps=5_000)

    if kd_tree:
        pass

    h5_file.close()


if __name__ == "__main__":
    main(dist_matrix=True, coordinates=False, kd_tree=False)
