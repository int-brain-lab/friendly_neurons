import numpy as np
import datajoint as dj
from ibl_pipeline import reference, subject, acquisition, behavior
from ibl_pipeline.analyses import behavior as behavior_analyses
from uuid import UUID
import datetime



def loading(eID, probe, region_list=[]):
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
        location= np.ndarray.tolist((histology.ClusterBrainRegionTemp() & key).fetch('acronym'))
    else: 
        command_list= region_commands(region_list)
        spikes_times = (ephys.DefaultCluster & key & (histology.ClusterBrainRegionTemp & (reference.BrainRegion() &  command_list))).fetch('cluster_spikes_times')
        location= np.ndarray.tolist((histology.ClusterBrainRegionTemp & key & (reference.BrainRegion() &  command_list)).fetch('acronym'))
    i=0
    clusters=[]
    for spike in spikes_times:
        clusters.append(np.full(len(spike), i, dtype=int))
        i+=1
    clusters=np.hstack(clusters)
    spikes=np.hstack(spikes_times)
    indices_sorted = np.argsort(spikes)
    spikes = spikes[indices_sorted]
    clusters=clusters[indices_sorted]
    trials=dict()
    trials["feedbackType"]=(behavior.TrialSet.Trial() & key).fetch('trial_feedback_type')
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

if __name__ == "__main__":
    exp_ID = "ecb5520d-1358-434c-95ec-93687ecd1396"
    brain_areas=["VIS"]
    print(loading(exp_ID, 0, brain_areas))
    print(loading(exp_ID, 1, brain_areas))
    print(loading(exp_ID, "both", brain_areas))