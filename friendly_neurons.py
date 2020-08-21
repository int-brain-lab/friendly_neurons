import json
from oneibl.one import ONE
import alf.io as ioalf
import ibllib.plots as iblplt
from pathlib import Path
import brainbox.io.one as bbone
import brainbox as bb
import numpy as np
from sklearn import manifold
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import numpy.ma as ma
import seaborn as sns
import collections
from pylab import *

### for community detection
import leidenalg as la
import igraph as ig
from igraph import *


from brainbox.processing import bincount2D
import matplotlib.pyplot as plt
import ibllib.plots as iblplt
from pathlib import Path


def community_detection(
    eID,
    probe="both",
    bin=0.02,
    sensitivity=1,
    visual=False,
    start="starts",
    end="ends",
):
    """
    Function:
    Takes an experiment ID and makes community detection analysis 


    Parameters:
    eID: experiment ID 
    probe: name of the probe wanted or both for both probes
    bin: the size of the bin
    sensitivity: the sensibility parameter for the leiden algorithm
    visual: a boolean on whether visualization is wanted
    starts: the name of the type of start intervals
    ends: the name of the type of end intervals


    Return:
    partition: ig graph vertex partition object
    partition_dictionary: a dictionary with keys for each community and sets as values with the indices of the clusters that belong to that community
    region_dict: similar to partition dictionary, but replaces each cluster with the region it belongs to
    
    Example:
    without a know path:
    >>>community_detection(
            exp_ID,
            visual=True,
            probe="probe00",
            start="stimOn_times",
            end="response_times",
        )
    with a known path "\\directory\\": 
        community_detection(
            exp_ID,
            visual=True,
            path="\\directory\\"
            probe="probe00",
            start="stimOn_times",
            end="response_times",
        )


    """

    spikes, clusters, trials, locations = loading_data(eID, probe)
    starts, ends = section_trial(start, end, trials)
    partition = community_detections_helper(
        spikes, clusters, starts, ends, bin, visual, sensitivity
    )
    partition_dictionary = dictionary_from_communities(partition)
    region_dict = location_dictionary(partition_dictionary, locations)
    return partition, partition_dictionary, region_dict


def section_trial(start, end, trials):
    """
    Function:
    The function connects with the database and gets the objects from that experiment


    Parameters:
    start: key of the dictionary in trials or start for the start of the interval
    end:key of the dictionary in trials or end for the end of the interval
    trials: dictionary from the trials object


    Return:
    starts: time series corresponding to the start of the interval
    ends:time series corresponding to the end of the interval

    """
    starts = None
    ends = None
    if start == "starts":
        starts = trials["intervals"][:, 0]
    elif start in trials:
        starts = trials[start]
    else:
        raise Exception("Non possible starts.")

    if end == "stops":
        ends = trials["intervals"][:, 1]
    elif end in trials:
        ends = trials[end]
    else:
        raise Exception("Non possible ends.")

    return starts, ends


def loading_data(eID, probe):
    """
    Function:
    The function connects with the database and gets the objects from that experiment


    Parameters:
    eID: the ID of the experiment
    probe:the probe for the analysis or the word "both" for both probes


    Return:
    spikes: an array with the times of all the spikes
    clusters: an array with a label for each cluster
    trials: a 'trial' object from the IBLLIB library
    locations: a list with the size of the number of clusters 

    """
    one = ONE()
    spikes, clusters, channels = bbone.load_spike_sorting_with_channel(eID, one=one)
    trials = one.load_object(eID, "trials")

    if probe == "both":

        spikes0 = spikes["probe00"]["times"]
        clusters0 = spikes["probe00"]["clusters"]
        location0 = clusters["probe00"]["acronym"]
        spikes1 = spikes["probe01"]["times"]
        clusters1 = spikes["probe01"]["clusters"]
        location1 = clusters["probe01"]["acronym"]
        location = location0 + location1
        num_clusters = np.max(clusters0) + 1 - np.min(clusters0)
        clusters1 += num_clusters
        i1 = 0
        i2 = 0
        spikes = []
        clusters = []
        while i1 + i2 < len(spikes0) + len(spikes1):
            if i1 < len(spikes0) and (i2 == len(spikes1) or spikes0[i1] < spikes1[i2]):
                spikes.append(spikes0[i1])
                clusters.append(clusters0[i1])
                i1 += 1
            else:
                spikes.append(spikes1[i2])
                clusters.append(clusters1[i2])
                i2 += 1
        spikes = np.array(spikes)
        clusters = np.array(clusters)
    else:
        location = clusters[probe]["acronym"]
        clusters = spikes[probe]["clusters"]
        spikes = spikes[probe]["times"]

    return spikes, clusters, trials, location


def dictionary_from_communities(partition):
    """
 

    Parameters:
    partition: VertexPartition type object


    Returns:

    community: a dictionary with keys for each community and sets as values with the vertices that belong to that community
    """
    community = dict()
    member = partition.membership
    vertices = [i.index for i in partition.graph.vs]
    for i in range(len(member)):
        if member[i] in community:
            community[member[i]].add(vertices[i])
        else:
            community[member[i]] = set([vertices[i]])
    return community


def location_dictionary(partition_dict, cluster_region):
    """
 

    Parameters:
    partition: partition dictionary with each community and the number of clusters

    Returns:

    community: a dictionary with keys for each community and sets as values with the vertices that belong to that community
    """
    region_dict = dict()
    for i in partition_dict:
        region_set = set()
        for j in partition_dict[i]:
            region_set.add(cluster_region[j])

        region_dict[i] = region_set
    return region_dict


def community_detections_helper(
    spikes, clusters, starts, ends, bins, visual, sensitivity
):
    """
    Function:
    cleaves the time array for the spikes such that only intervals of interest are 
    compiled into a time series array from which a correlation is gotten. 
    From this correlation a graph is made on which the Leiden community detection is run. 


    Parameters:
    spikes: array with times for all the spikes
    clusters: array with the names of the clusters
    starts: array with stars of the intervals considered for the spikes
    ends: array with ends of the intervals considered for the spikes
    bins: the size of the bins
    visual: boolean that marks whether visuals are wanted



    Return:
    partition: "vertexpartition" object from the igraph library


    """

    def visualize():
        """
        Function:
        Plots graphs for all intermediate data.
    ````Parameters:
        None
        Return 
        None



        """
        # cluster = [13, 53, 270]
        # for i in cluster:
        #    plt.plot(spikes_matrix[i, :])
        #    plt.title("Spike count for cluster {}".format(i))
        #    plt.xlabel("bins ({} s each)".format(bins))
        #    plt.ylabel("number of spikes")
        #    plt.show()

        # sns.heatmap(correlation_matrix_original, square=True)
        # plt.title("Trial duration correlations (stim onset to reaching target)")
        # plt.show()

        visual_style1 = {}
        f = lambda x: x if x > 0 else 0
        visual_style1["edge_width"] = [f(w) * 0.25 for w in neuron_graph.es["weight"]]
        visual_style1["layout"] = "circle"
        visual_style1["labels"] = True
        visual_style1["vertex_size"] = 20
        visual_style1["vertex_color"] = "moccasin"
        plot(neuron_graph, **visual_style1)
        visual_style = {}
        f = lambda x: x if x > 0 else 0
        visual_style["edge_width"] = [f(w) * 0.25 for w in neuron_graph.es["weight"]]
        visual_style["layout"] = "circle"
        visual_style["labels"] = True
        visual_style["vertex_size"] = 20
        plot(partition, **visual_style)

    spikes_interval, clusters_interval = interval_selection(
        spikes, clusters, starts, ends
    )

    spikes_matrix = bb.processing.bincount2D(
        spikes_interval, clusters_interval, xbin=bins
    )[0]
    correlation_matrix_original = np.corrcoef(spikes_matrix)
    correlation_matrix = correlation_matrix_original[:, :]
    correlation_matrix[correlation_matrix < 0] = 0
    np.fill_diagonal(correlation_matrix, 0)
    neuron_graph = ig.Graph.Weighted_Adjacency(
        correlation_matrix.tolist(), mode=ADJ_UNDIRECTED
    )
    neuron_graph.vs["label"] = [f"{i}" for i in range(np.max(clusters))]
    if not sensitivity == 1:
        partition = la.RBConfigurationVertexPartition(
            neuron_graph, resolution_parameter=sensitivity
        )
        optimiser = la.Optimiser()
        optimiser.optimise_partition(partition)
    else:
        partition = la.find_partition(neuron_graph, la.ModularityVertexPartition)

    partition = la.find_partition(neuron_graph, la.ModularityVertexPartition)

    visualize() if visual else None

    return partition


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
    if len(starts) == len(ends):
        temp_x = []
        temp_y = []
        for i in range(len(starts)):
            temp_indices = np.where((x >= starts[i]) & (x <= ends[i]))[0]
            temp_x = np.concatenate((temp_x, x[temp_indices]))
            temp_y = np.concatenate((temp_y, y[temp_indices]))
        return temp_x, temp_y
    else:
        raise Exception("starts and ends have to be the same length")


def main():
    print("communtity detection")


if __name__ == "__main__":
    exp_ID = "ecb5520d-1358-434c-95ec-93687ecd1396"
    path_e = "C:\\Users\\mitadm\\OneDrive - Massachusetts Institute of Technology\\MIT\\Summer\\Network UROP\\ibllib\\ibllib\\churchlandlab\\Subjects\\CSHL051\\2020-02-05\\001\\alf\\"

    print(
        community_detection(
            exp_ID, visual=True, probe="probe00", start="starts", end="stimOn_times",
        )[2]
    )
    print(
        community_detection(
            exp_ID, visual=True, probe="both", start="starts", end="stimOn_times",
        )[2]
    )
    print(
        community_detection(
            exp_ID,
            visual=True,
            probe="probe00",
            start="stimOn_times",
            end="response_times",
        )[2]
    )
    print(
        community_detection(
            exp_ID, visual=True, probe="probe00", start="response_times", end="stops",
        )[2]
    )
