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
    locations: a list of the locations for each cluster
    
    
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
    if not bool(data):
        spikes, clusters, trials, locations = loading_DataJoint( eID , probe, region_list)
    else: 
        spikes, clusters, trials, locations = data
        
    starts, ends = section_trial(user_start, user_end, trials, feedbackType)
    partition = community_detections_helper(
        spikes, clusters, starts, ends, bin, visual, sensitivity
    )
    partition_dictionary = dictionary_from_communities(partition)
    return partition, partition_dictionary, locations

def brain_region( eID,
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


    partition, partition_dictionary, locations= community_detection(eID,probe,bin,sensitivity,visual,feedbackType,user_start,user_end,data=data,region_list=region_list)
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


def loading_DataJoint(eID, probe, region_list=[]):
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
    ephys = dj.create_virtual_module('ephys', 'ibl_ephys')
    histology = dj.create_virtual_module('histology', 'ibl_histology')
    key_session=[{'session_uuid': UUID(eID)}]
    key = (acquisition.Session & ephys.DefaultCluster & key_session).fetch('KEY', limit=1)
    if probe !='both':
        key[0]['probe_idx']=probe
    if len(region_list)==0: 
        spikes_times = (ephys.DefaultCluster & key).fetch('cluster_spikes_times')
        location= ndarray.tolist((histology.ClusterBrainRegionTemp() & key).fetch('acronym'))
    else: 
        command_list= region_commands(region_list)
        spikes_times = (ephys.DefaultCluster & key & (histology.ClusterBrainRegionTemp & (reference.BrainRegion() &  command_list))).fetch('cluster_spikes_times')
        location= ndarray.tolist((histology.ClusterBrainRegionTemp & key & (reference.BrainRegion() &  command_list)).fetch('acronym'))
    i=0
    clusters=[]
    for spike in spikes_times:
        clusters.append(np.full(len(spike), i, dtype=int))
        i+=1
    clusters=np.hstack(clusters)
    spikes=np.hstack(spikes_times)
    indices=[l for l in range(len(clusters))]
    indices_sorted=sorted(indices, key=lambda x: spikes[x])
    indices=np.array(indices_sorted)
    clusters=clusters[indices_sorted]
    
    trials=dict()
    trials["feedbackType"]=(behavior.TrialSet.Trial() & key).fetch('trial_feedback_type')
    trials["intervals"]=np.transpose(np.vstack([(behavior.TrialSet.Trial() & key).fetch('trial_start_time'),(behavior.TrialSet.Trial() & key).fetch('trial_end_time')]))
    trials["stimOn_times"]=(behavior.TrialSet.Trial() & key).fetch('trial_stim_on_time')
    trials["response_times"]=(behavior.TrialSet.Trial() & key).fetch('trial_response_time')

    return spikes, clusters, trials, location


def region_commands(s_list):
    result=[]
    for region in s_list:
        result.append('acronym like "%'+region+'%"')
    return result

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

    spikes_interval, clusters_interval = interval_selection(
        spikes, clusters, starts, ends
    )

    spikes_matrix = bb.processing.bincount2D(
        spikes_interval, clusters_interval, xbin=bins
    )[0]
    spikes_matrix_fixed=addition_of_empty_neurons(spikes_matrix,clusters,clusters_interval)
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
        # partition = la.CPMVertexPartition(
        #    neuron_graph, resolution_parameter=sensitivity
        # )

        optimiser = la.Optimiser()
        optimiser.optimise_partition(partition)
    else:
        partition = la.find_partition(neuron_graph, la.ModularityVertexPartition)

    partition = la.find_partition(neuron_graph, la.ModularityVertexPartition)

    visualize() if visual else None

    return partition

def addition_of_empty_neurons(spikes_matrix,clusters,clusters_interval):
    """


    """

    present_clusters=[int(i) for i in set(clusters_interval)]
    present_clusters_set=set(present_clusters)
    present_clusters.sort()

    n_clusters= int(max(clusters))+1
    k_spikes=len(spikes_matrix[0][:])
    spikes_matrix_fixed=None
    k=0
    for j in range(n_clusters):
        if j==0:
            if j in present_clusters_set: 
                spikes_matrix_fixed=np.array([spikes_matrix[k][:]])
                k+=1
            else: 
                spikes_matrix_fixed=np.array([np.zeros(k_spikes)])
        else:
            if j in present_clusters_set: 
                spikes_matrix_fixed=np.concatenate((spikes_matrix_fixed,np.array([spikes_matrix[k][:]])))
                k+=1
            else: 
                spikes_matrix_fixed=np.concatenate((spikes_matrix_fixed,np.array([np.zeros(k_spikes)])))
    return spikes_matrix_fixed





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
    brain_areas=["VIs"]
    


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
