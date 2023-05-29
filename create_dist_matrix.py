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
def write_dist_matrix_to_hdf5_via_multiprocessing(h5_file):
    group = h5_file.create_group(f'{MODE}_dist_matrix')

    start = time.time()

    # initialization
    start_idx = 0  # 342_000
    end_idx = 5000  # 342_042
    steps = 5000

    while start_idx < end_idx:
        if (end_idx - start_idx) < steps:
            steps = end_idx - start_idx

        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            iter_steps = list(range(start_idx, start_idx + steps))  # results are in correct order

            results = pool.map(compute_geodesic_distance, iter_steps)

        # group.create_dataset(f'Foo', data=results) # to store as a whole matrix
        for idx in iter_steps:
            group.create_dataset(f'{idx}', data=results[idx - start_idx])  # rm offset

        start_idx += steps

    diff = time.time() - start
    print('-' * 60)
    print(Colors.OKBLUE + f'{diff} secs = {diff / 60} mins' + Colors.ENDC)  # for readability
    print('-' * 60)


def hdf5_file_exists(path):
    if os.path.isfile(path):

        print(Colors.WARNING + 'Warning: File already exists.' + Colors.ENDC)

        choice = input('Do you want to continue (C) or stop (S) the code? ')

        if choice != 'C':
            exit()


def main(dist_matrix=False, coordinates=True, kd_tree=False):
    global solver  # to make it work with multiprocessing
    global MODE  # to make it work with multiprocessing

    MODE = 'potpourri3d'
    path = './geodesic_distance.h5'  # store hdf5 file here

    hdf5_file_exists(path=path)
    h5_file = h5py.File(path, 'a')  # append additional group

    if coordinates:
        vertices, _ = get_space()
        h5_file.create_dataset('coordinates', data=vertices)  # float64

    if dist_matrix:
        vertices, faces = get_space()
        solver = init_solver(vertices=vertices, faces=faces)
        write_dist_matrix_to_hdf5_via_multiprocessing(h5_file=h5_file)

    if kd_tree:
        pass

    h5_file.close()


if __name__ == "__main__":
    main(dist_matrix=True, coordinates=False, kd_tree=False)
