
import numpy as np





def interval_selection(x, y, starts, ends):
    """
    Function:
    Chops the intervals in the x and y variable arrays

    Parameters:
    x: x variable array
    y: y variable array
    starts: starts of intervals in x
    ends: ends of intervals in y
    bins: size of the bins

    Return:
    temp_x=x variable array with the elements not belonging to the variable array
    temp_y= y variable array with the elements not belonging to the variable array

    """


    idx_stim = x * 0
    idx_stim[np.searchsorted(x, starts)] = 1
    idx_stim[np.searchsorted(x, ends)] = -1
    idx_stim = np.cumsum(idx_stim).astype(np.bool)
    temp_x2 = x[idx_stim]
    temp_y2 = y[idx_stim]

    return temp_x2, temp_y2

def addition_of_empty_neurons(spikes_matrix,clusters,clusters_interval):
    """
    This function adds empty entries for clusters that have no activity

    Parameters:

    Return:
    spikes_matrix_fixed: matrix with filled entries for clusters

    """
    n_clust = np.max(clusters) + 1 #or if we get it from datajoint earlier safer
    included_clust = np.unique(clusters_interval).astype(int)
    spikes_matrix_fixed = np.zeros((n_clust, spikes_matrix.shape[1]))
    spikes_matrix_fixed[included_clust, :] = spikes_matrix
    return spikes_matrix_fixed


if __name__ == "__main__":
    exp_ID = "ecb5520d-1358-434c-95ec-93687ecd1396"
    brain_areas=["VIS"]
    