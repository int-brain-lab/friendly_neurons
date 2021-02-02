import json

import alf.io as ioalf
import ibllib.plots as iblplt
import brainbox.io.one as bbone
import brainbox as bb
import numpy as np
import numpy.ma as ma

import collections
from pylab import *
### for community detection
import leidenalg as la
import igraph as ig
from igraph import *

### for dat analysis 
from brainbox.processing import bincount2D
import matplotlib.pyplot as plt
import ibllib.plots as iblplt
### DataJoint
import datajoint as dj
from ibl_pipeline import reference, subject, acquisition, behavior
from ibl_pipeline.analyses import behavior as behavior_analyses
from uuid import UUID
import datetime

###file_dependencies
import spikes_processing as spp
import dj_loading as djl


def community_detection(
    eID,
    probe="both",
    bin=0.02,
    sensitivity=1,
    visual=False,
    feedbackType=None,
    user_start="trial_start",
    user_end="trial_end",
    region_list=[],
    data=None
    
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
    feedbackType: value for feedback wanted
    starts: the name of the type of start intervals
    ends: the name of the type of end intervals



    Return:
    partition: ig graph vertex partition object
    partition_dictionary: a dictionary with keys for each community and sets as values with the indices of the clusters that belong to that community, and the key
    region_dict: dictionary keyed by community number and value of a dictionary with the names of the brain regions of that community and their frequency
    locations: a list of the locations for each cluster
    
    
    Example:
    without a know path:
    >>>community_detection(
    >>community_detection(
            exp_ID,
            visual=True,
            probe="probe00",
            start="stimOn_times",
            end="response_times",
        )s
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
    if not bool(data):
        spikes, clusters, trials, locations = djl.loading( eID , probe, region_list)
    else: 
        spikes, clusters, trials, locations = data
        
    starts, ends = section_trial(user_start, user_end, trials, feedbackType)

    spikes_interval, clusters_interval = spp.interval_selection(
        spikes, clusters, starts, ends
    )

    spikes_matrix = bb.processing.bincount2D(
        spikes_interval, clusters_interval, xbin=bin, # xlim=[0, nclusters]
    )[0]
    spikes_matrix_fixed=spp.addition_of_empty_neurons(spikes_matrix,clusters,clusters_interval)
    correlation_matrix_original = np.corrcoef(spikes_matrix_fixed)
    correlation_matrix = correlation_matrix_original[:, :]
    correlation_matrix[correlation_matrix < 0] = 0
    np.fill_diagonal(correlation_matrix, 0)
    neuron_graph = ig.Graph.Weighted_Adjacency(
        correlation_matrix.tolist(), mode="UNDIRECTED"
    )
    neuron_graph.vs["label"] = [f"{i}" for i in range(np.max(clusters))]
    
    if sensitivity != 1:

        partition = la.RBConfigurationVertexPartition(
            neuron_graph, resolution_parameter=sensitivity
        )
        optimiser = la.Optimiser()
        optimiser.optimise_partition(partition)
    else:
        partition = la.find_partition(neuron_graph, la.ModularityVertexPartition)

    visualization(neuron_graph, partition) if visual else None
    partition_dictionary = dictionary_from_communities(partition)
    region_dict = location_dictionary(partition_dictionary, locations)
    return partition, partition_dictionary, region_dict, locations


def section_trial(user_start, user_end, trials, feedbackType):
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
    if user_start == "trial_start":
        starts = trials["intervals"][:, 0]
    elif user_start in trials:
        starts = trials[user_start]
    else:
        raise Exception("Non possible starts.")

    if user_end == "trial_end":
        ends = trials["intervals"][:, 1]
    elif user_end in trials:
        ends = trials[user_end]
    else:
        raise Exception("Non possible ends.")
    if feedbackType == None:
        return starts, ends
    else:
        fbTypes = trials["feedbackType"]
        temp_indices = np.where(fbTypes == feedbackType)[0]
        starts = starts[temp_indices]
        ends = ends[temp_indices]
        return starts, ends


def visualization(neuron_graph, partition):
        """
        Function:
        Handles visualization of the graphs
        Parameters:
        neuron_graph=matrix represetation of the graph
        partition= partition object from igraph
        Return 
        ----

        """
        visual_style1 = {}
        edge_darkness=0.5
        f = lambda x: x if x > 0 else 0
        visual_style1["edge_width"] = [f(w) *  edge_darkness for w in neuron_graph.es["weight"]]
        visual_style1["layout"] = "circle"
        visual_style1["labels"] = True
        visual_style1["vertex_size"] = 20
        visual_style1["vertex_color"] = "moccasin"
        plot(neuron_graph, **visual_style1)
        visual_style = {}
        f = lambda x: x if x > 0 else 0
        visual_style["edge_width"] = [f(w) *edge_darkness  for w in neuron_graph.es["weight"]]
        visual_style["layout"] = "circle"
        visual_style["labels"] = True
        visual_style["vertex_size"] = 20
        plot(partition, **visual_style)



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
    regions_dict = dict()
    for i in partition_dict:
        section_dict = dict()
        for j in partition_dict[i]:
            temp_name=cluster_region[j]
            if temp_name in section_dict:
                section_dict[temp_name]+=1
            else:
                section_dict[temp_name]=1
           

        regions_dict[i] = section_dict
    return regions_dict


def main():
    print("communtity detection")


if __name__ == "__main__":
    exp_ID = "ecb5520d-1358-434c-95ec-93687ecd1396"
    brain_areas=["VIS"]
    


    ### Single Probe Three Time Periods ###
    print(
        community_detection(
            exp_ID,
            visual=True,
            probe=0,
            user_start="trial_start",
            user_end="stimOn_times",
        )
    )

    print(
        community_detection(
            exp_ID,
            visual=True,
            probe=0,
            user_start="trial_start",
            feedbackType=1,
            user_end="stimOn_times",
        )
    )
    print(
        community_detection(
            exp_ID,
            visual=True,
            probe=0,
            user_start="trial_start",
            feedbackType=-1,
            user_end="stimOn_times",
        )
    )

    ### Both Probes StimOnset ###


    print(
        community_detection(
            exp_ID,
            visual=True,
        
            user_start="trial_start",
            user_end="stimOn_times",
        )
    )
    print(
        community_detection(
            exp_ID,
            visual=True,
            user_start="stimOn_times",
            user_end="response_times",
        )
    )
    print(
        community_detection(
            exp_ID,
            visual=True,
            user_start="response_times",
            user_end="trial_end",
        )
    )

    ### Both Probes Three Time Periods Visual Area ###
    brain_areas=["VIS"]
    print(
        community_detection(
            exp_ID,
            visual=True,
            probe="both",
            user_start="trial_start",
            user_end="stimOn_times",
            region_list=brain_areas
        )
    )
    print(
        community_detection(
            exp_ID,
            visual=True,
            user_start="stimOn_times",
            user_end="response_times",
            region_list=brain_areas
        )
    )
    print(
        community_detection(
            exp_ID,
            visual=True,
            user_start="response_times",
            user_end="trial_end",
            region_list=brain_areas
        )
    )
