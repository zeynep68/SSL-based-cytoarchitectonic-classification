import unittest

import h5py
import numpy as np

# custom libraries
from create_dist_matrix import get_space


class Tests(unittest.TestCase):
    def test_saved_hdf5_file_is_correct(self):
        """ Test if created file contains the desired values from the library. """
        vertices, _ = get_space()

        test_file = 'geodesic_distance.h5'

        with h5py.File(test_file, 'r') as h5_file:
            # check if data points exist

            assert isinstance(h5_file['coordinates'], h5py.Dataset)

            assert h5_file['coordinates'].shape == (342_042, 3)

            assert np.array_equal(h5_file['coordinates'][:], vertices, equal_nan=True)

    def test_sample_wise_diag_is_zero(self):
        test_file = 'geodesic_distance.h5'

        with h5py.File(test_file, 'r') as h5_file:
            group = h5_file['potpourri3d_dist_matrix']

            rand_indices = np.random.choice(list(group.keys()), 100)

            for idx in rand_indices:
                assert group[idx][int(idx)] == 0

    def test_sample_wise_pairwise_distances(self):
        test_file = 'geodesic_distance.h5'

        with h5py.File(test_file, 'r') as h5_file:
            group = h5_file['potpourri3d_dist_matrix']
            """
            rand_indices = np.random.choice(list(group.keys()), 10)
            vertices, faces = get_space()
            solver = init_solver(vertices=vertices, faces=faces)
            for idx in rand_indices:
                distances = group[idx]
                assert distances[4] == 0
            """

    def test_first_last_row_exists_in_dist_matrix(self):
        """ The rows of the distance matrix are inserted in order. Check for
         the first and last point's distances to all the other points. """
        return


"""
def test_foo():
    # sample random indices
    # pick corresponding row
    # and compare this row
    # to pairwise computation to check each value is same
    return


d
"""

if __name__ == "__main__":
    unittest.main()
