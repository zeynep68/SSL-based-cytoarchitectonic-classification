import h5py
import numpy as np

"""
Dist_matrix with shape: (48251, 48251)
mean : 127.5
std  : 39.57909487883818

"""

def min_max_normalization(coords, new_min=-1., new_max=1., h5_file=None):
    old_min = np.min(coords, axis=0)
    old_max = np.max(coords, axis=0)

    diff = old_max - old_min

    new_coords = ((coords - old_min) / diff) * (new_max - new_min) + new_min

    if h5_file is not None:
        h5_file.create_dataset('min_max_normalized_coords', data=new_coords)

    return new_coords


def compute_std(dist_matrix, mean, verbose=True):
    if mean is None:
        return np.std(dist_matrix)

    std = 0

    num_rows, num_cols = dist_matrix.shape
    N = num_rows * num_cols

    for i in range(num_rows):
        row = dist_matrix[i, :]
        std += np.sum( ((row - mean)**2) / N )

    if verbose:
        print('Standard deviation:', np.sqrt(std))

    return np.sqrt(std)


if __name__ == "__main__":
    h5_file = h5py.File('geodesic_distance.h5', 'a')

    dist_matrix = h5_file['dist_matrix'][:]

    mean = np.mean(dist_matrix)
    std = compute_std(dist_matrix, mean)

    normalized_dist_matrix = (dist_matrix - mean) / std

    h5_file.create_dataset('standardized_dist_matrix', data=normalized_dist_matrix)

