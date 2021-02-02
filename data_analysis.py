


from ibl_pipeline import subject, acquisition, ephys, histology
from tqdm import tqdm
import datajoint as dj
import numpy as np 
import matplotlib.pyplot as plt
import random
from uuid import UUID 
import datetime
import igraph as ig 
import pickle


###file dependencies


from friendly_neurons import community_detection
from  dj_loading import loading



def general_analysis(
    exp_ID,probe, 
    visual=False, 
    height=1000, 
    length=1000,
    feedbackType=None, 
    base_layout=1, 
    timeperiods=[("trial_start","stimOn_times"), ("stimOn_times", "response_times"), ("response_times", "trial_end") ], 
    region_list=[]):
    """

    Makes a general analysis of th three main time perios (prestimulus, during stimulus, and after the stimulus)


    Input:
    exp_ID: experiment ID 
    visual: flag for visuals
    height: height of the resulting graph
    length: length of the resulting graph
    feedBackType: the type of response wanted
    base_layout: index in timeperiods of the base_layout
    timeperiods: names of time series used for reach graph



    Output:


    None

    """
    

    ###Data collection 
    results=[]
    
    data=loading( exp_ID , probe,region_list=region_list)
    for time1, time2 in timeperiods:
        temp_dict=dict()
        graph, partition, regions, locations=community_detection(exp_ID, visual=visual, probe=probe, user_start=time1, user_end=time2, feedbackType=feedbackType,data=data, region_list=region_list)
        temp_dict["graph"]=graph
        temp_dict["partition"]=partition
        temp_dict["regions"]= regions
        temp_dict["locations"]= locations
        results.append(temp_dict)

  
    ###Analysis functions


    ###Summary of each partition

    """
    Here we provide different measurements.

    Summary: this is just a breakdown of the community alligeance 
    Split Join Distance: Compares the partitions between time periods 
    Compare Communities: computes information variance for each community 
    Ovelap of communities: computes the percentage overlap with respect to the communities of the base layout


    """
    print("Summaries")
    for i in range(len(timeperiods)):
        time1, time2 = timeperiods[i]
        print(time1+" to "+ time2)
        print(results[i]["graph"].summary(verbosity=1))

    ###Split Join Distance 
    
    print("Split join distance")
    #print([[ split_join_distance(graphs_for_partitions[i],graphs_for_partitions[j] ) for i in range(len(graphs_for_partitions))] for j in range(len(graphs_for_partitions))])
    
    ###Compare Communities
    
    print("Compare communities")

    print([[ ig.compare_communities(i["graph"],j["graph"] ) for i in results] for j in results])


    print("Ovelap of communities")
    results_no_base= [j for j in results]
    results_no_base= results_no_base[:base_layout]+results_no_base[base_layout+1:]
    partitions_no_base= [ i["partition"]  for i in results_no_base] 
    overlaps=com_overlap(results[base_layout]["partition"], partitions_no_base)
    print(overlaps)


    ###Community assignments based on percentage shared with original community

    print("Matchings ")
    matchings=[] 
    for i in range(len(results_no_base)):
        matchings.append(match_colors( overlaps[i], len(results[base_layout]["partition"]) , len(results_no_base[i]["partition"])))
    print(matchings)

    ###Locations by community
    locations_simplified=[parse(i) for i in locations]


    partition=results[base_layout]["partition"]
    num_clusters= max([ max(partition[i]) for i in partition])+1
    num_to_map=dict()
    for  i in range(num_clusters):
        num_to_map[i]={num_clusters-1-i} 

    
    locations_to_order=[ i for i in set(locations_simplified)]
    locations_to_order.sort()
    print(locations_to_order)
    order_x=labels_to_dictionary(locations_to_order)
    for i in order_x:
        temp=0
        for k in order_x[i]:
            temp=k
        order_x[i]=temp

    layout_x=locations_from_dictionary(partition,length=length, height=height, seperated=False)
    layout_y=locations_from_dictionary(labels_to_dictionary(locations_simplified),order=order_x , length=length, height=height)
    layout_probe_y=locations_from_dictionary(num_to_map, length=length, height=height)
    layout_probe_3= [ [ int(length*( 1/2-0.1+0.2*i%3)),layout_probe_y[i][1]] for i in range(len(layout_probe_y))]
    layout_mixed=[ [layout_x[i][0],layout_y[i][1]] for i in range(min(len(layout_y),len(layout_x)))]
    layout_depth=[ [layout_x[i][0],layout_probe_y[i][1]] for i in range(min(len(layout_probe_y),len(layout_x)))]
    coloring= ig.ClusterColoringPalette(20)
    colorings=[]
    for i in range(len(overlaps)):
        o_partition=results_no_base[i]["partition"]
        overlap=overlaps[i]
        plt.table(cellText=[ [ overlap[(i,j)] for j in o_partition] for i in partition], rowLabels=[i for i in partition] , colLabels=[j for j in o_partition], loc='top' )
        plt.subplots_adjust(left=0.2, top=0.8)
        plt.show()
       
        colorings.append(Pallete_changed(20, coloring, matchings[i] ))
    
    
    ###Final visualization


    for i in range(len(timeperiods)):
        if i < base_layout: 
            pre_graph=results[i]["graph"]
            visualize(pre_graph, layout= layout_mixed , vertex_size=30, labels=locations_simplified, length=length, height=height, coloring=colorings[i])
            visualize(pre_graph, layout= layout_probe_3 , vertex_size=30, labels=locations_simplified, length=length, height=height, coloring=colorings[i])
            visualize(pre_graph, layout= layout_depth , vertex_size=30, labels=locations_simplified, length=length, height=height, coloring=colorings[i])
        elif i==base_layout:
            pre_graph=results[i]["graph"]
            visualize(pre_graph, layout= layout_mixed , vertex_size=30, labels=locations_simplified, length=length, height=height, coloring=coloring)
            visualize(pre_graph, layout= layout_probe_3 , vertex_size=30, labels=locations_simplified, length=length, height=height, coloring=coloring)
            visualize(pre_graph, layout= layout_depth , vertex_size=30, labels=locations_simplified, length=length, height=height, coloring=coloring)

        else: 
            j=i-1
            pre_graph=results[i]["graph"]
            visualize(pre_graph, layout= layout_mixed , vertex_size=30, labels=locations_simplified, length=length, height=height, coloring=colorings[j])
            visualize(pre_graph, layout= layout_probe_3 , vertex_size=30, labels=locations_simplified, length=length, height=height, coloring=colorings[j])
            visualize(pre_graph, layout= layout_depth , vertex_size=30, labels=locations_simplified, length=length, height=height, coloring=colorings[j])
    
    return visualized_3d(results, matchings, base_layout,exp_ID, probe)
   

class Pallete_changed(ig.ClusterColoringPalette):
    """

    Altered pallete that makes an assignment based on the matching dictionary if the color is already in the dictionary


    """

    def __init__(self, n,  pallete=None, change_dict=None):
        self.prev_pallete=pallete
        self.dict=change_dict
    

    def _get(self, v):
        """
        Creates a new pallete
        """
        return self.prev_pallete.get(self.dict[v])
    
    def get(self, v):
        """
        Creates a new pallete
        """
        if  v in self.dict:
            return self.prev_pallete.get(self.dict[v])
        else:
            return self.prev_pallete.get(v)



def match_colors(overlap_dict, n, m, threshold=0.2):
    """

    Based on the overlap from a series of dictionaries 

    uses 

    """


    start=set([(i, j) for i in range(n) for j in range(m)])
    result_dict={}

    while len(start)>0: 
        max_assign=max(start, key=lambda f: overlap_dict[f])
        result_dict[max_assign[1]]= max_assign[0]

        for i in range(n):
            try:
                start.remove((i, max_assign[1]))
            except:
                pass
        for j in range(m):
            try:
                start.remove((max_assign[0],j))
            except:
                pass

    for i in range(m):
        if i not  in result_dict:
            result_dict[i]=n+i
        else:
            if threshold> overlap_dict[( result_dict[i] ,i)]:
                result_dict[i]=n+i
    return result_dict
        
            
def match_colors_old(overlap_dict, n, m):
    """

    Based on the overlap from a series of dictionaries 

    """


    start=set([(i, j) for i in range(n) for j in range(m)])
    result_dict={}

    while len(start)>0: 
        max_assign=max(start, key=lambda f: overlap_dict[f])
        result_dict[max_assign[1]]= max_assign[0]

        for i in range(n):
            try:
                start.remove((i, max_assign[1]))
            except:
                pass
        for j in range(m):
            try:
                start.remove((max_assign[0],j))
            except:
                pass

    for i in range(m):
        if i not  in result_dict:
            result_dict[i]=n+i
        
    return result_dict



    


def com_overlap(metric, other_communities): 
    """
    compares communities in other_communities with the metric community to check overlap 
    """
    results=[]
    for partition in other_communities:
        temp=dict()
        for j in metric: 
            for k in partition:
                temp[(j,k)]=len(set(metric[j]).intersection(set(partition[k])))/len(metric[j])
        results.append(temp)
    return results 
    

def com_overlap_new(metric, other_communities): 
    """
    compares communities in other_communities with the metric community to check overlap 
    """
    results=[]
    for partition in other_communities:
        temp=dict()
        for j in metric: 
            for k in partition:
                temp[(j,k)]=len(set(metric[j]).intersection(set(partition[k])))/len(set(partition[k]))
        results.append(temp)
    return results 





def communities_nice_outline(pre_graph,graph, post_graph, pre_partition, post_partition, length, height, locations_simplified, order_x,  name=""):
    layout_x=locations_from_dictionary(pre_partition,length=length, height=height, seperated=False)
    layout_y=locations_from_dictionary(labels_to_dictionary(locations_simplified),order=order_x , length=length, height=height)
    layout_mixed=[ [layout_x[i][0],layout_y[i][1]] for i in range(min(len(layout_y),len(layout_x)))]

    visualize(pre_graph, layout= layout_mixed, vertex_size=40, labels=locations_simplified, length=length, height=height)
    visualize(graph, layout= layout_mixed, vertex_size=40, labels=locations_simplified, length=length, height=height)
    visualize(post_graph, layout= layout_mixed, vertex_size=40, labels=locations_simplified, length=length, height=height)




def locations_from_dictionary(dictionary, order=None, length=600, height=600, seperated=True):
    random.seed()
    num_clusters= max([ max(dictionary[i]) for i in dictionary])
    positions= [ 0 for j in range(num_clusters+1)]
    if order==None:
        max_size=max(dictionary)+1
    else: 
        max_size=len(dictionary.keys())
    for i in dictionary: 
        for j in dictionary[i]:
            if order==None:
                if seperated: 
                    positions[j]=[int(length*random.uniform((i+0.25)/max_size , (i+0.75)/max_size)),int(height*random.uniform((i+0.25)/max_size , (i+0.75)/max_size))]
                else:
                    positions[j]=[int(length*random.uniform((i)/max_size , (i+1)/max_size)),int(height*random.uniform((i)/max_size , (i+1)/max_size))]
            else:
                if seperated: 
                    positions[j]=[int(length*random.uniform((order[i]+0.25)/max_size , (order[i]+0.75)/max_size)),int(height*random.uniform((order[i]+0.25)/max_size , (order[i]+0.75)/max_size))]
                else:
                    positions[j]=[int(length*random.uniform((order[i])/max_size , (order[i]+1)/max_size)),int(height*random.uniform((order[i])/max_size , (order[i]+1)/max_size))]

    return positions

def labels_to_dictionary(x):
    result=dict()
    for i in range(len(x)):
        if x[i] in result:
            result[x[i]].add(i)
        else:
            result[x[i]]=set([i])
    return result 




def parse(word):
    if word[0].isupper():
        temp_word=""
        i=0
        while i< len(word) and  word[i].isupper():
            temp_word+=word[i]
            i+=1
        return temp_word

    else:
        return word 

def sections(partition):
    """
    makes a dictionary that returns each section of a partition with a number o

    """
    temp_dict=dict()
    for i in partition:
        temp_dict[i]=dict()
        for j in partition[i]:
            temp=parse(j)
            
            if temp in temp_dict[i]:
                temp_dict[i][temp]+=partition[i][j]
            else:
                temp_dict[i][temp]=partition[i][j]
    return temp_dict

def belong_dictionary(partitions, matchings=None , base_layout=None ):
    """

    Makes a dictionary with pairs (i,j ) 


    """
    flag= (matchings is not None) & (base_layout is not None )
    result=dict()
    for i in range( len(partitions)): 
        flag_base= flag  & i != base_layout
        if flag_base:
            match= matchings[i]
        
        for j in partitions[i]: 
            for k in partitions[i][j]:
                if k in result:
                    if flag_base:
                        result[k].append(match[j])
                    else: 
                        result[k].append(j)
                else:
                    if flag_base:
                        result[k]=[match[j]]
                    else: 
                        result[k]=[j]

    return result

def visualize(g, layout="circle", vertex_size=20, labels=None, length=1000, height=1000, coloring=None  ):
    """
        Function:
        Plots graphs for all intermediate data.
    Parameters:
        None
        Return 
        None



     """
     

    visual_style1 = {}
    f = lambda x: x if x > 0 else 0
    visual_style1["edge_width"] = [f(w)  for w in g.graph.es["weight"]]
    if labels!=None:
        g.graph.vs["label"]=labels
    visual_style1["layout"] = layout
    visual_style1["labels"] = True
    visual_style1["bbox"]= (length, height)
    visual_style1["vertex_size"] = vertex_size
    if coloring!=None:
        visual_style1["palette"]=coloring
    ig.plot(g, **visual_style1)






def visualized_3d(results, matchings,base_layout, eID, probe):
    ephys = dj.create_virtual_module('ephys', 'ibl_ephys')
    histology = dj.create_virtual_module('histology', 'ibl_histology')
    key_session=[{'session_uuid': UUID(eID)}]
    key = (acquisition.Session & ephys.DefaultCluster & key_session).fetch('KEY', limit=1)
    
    if probe !='both':
        key[0]['probe_idx']=probe
    clusters = (ephys.DefaultCluster & key & histology.ChannelBrainLocationTemp).fetch('KEY')
    cluster_coords = []
    for key in tqdm(clusters):
        channel_raw_inds, channel_local_coordinates = \
            (ephys.ChannelGroup & key).fetch1(
                'channel_raw_inds', 'channel_local_coordinates')
        channel = (ephys.DefaultCluster & key).fetch1('cluster_channel')
        if channel in channel_raw_inds:
            channel_coords = (np.squeeze(
                channel_local_coordinates[channel_raw_inds == channel]))
            
            # get the Location with highest provenance
            q = histology.ChannelBrainLocationTemp & key & \
                dict(channel_lateral=channel_coords[0],
                    channel_axial=channel_coords[1]) & 'provenance=70'
            if q:
                cluster_coords.append(q.fetch('channel_x', 'channel_y', 'channel_z'))
            else:
                cluster_coords.append([0,0,0])
        else:
            cluster_coords.append([0,0,0])
    matchings.insert(base_layout, {} )
    allegiance=belong_dictionary([i["partition"] for i in results],matchings, base_layout)
    location= results[0]["locations"]
    clusters=len(cluster_coords)
    final=[]
    for i in range(clusters):
        final.append(cluster_coords[i]+[location[i]]+allegiance[i])
    return final 
        
        










if __name__ == "__main__":
    #expIDs=[["ecb5520d-1358-434c-95ec-93687ecd1396", "Anne"], ['c9fec76e-7a20-4da4-93ad-04510a89473b', "nate"], ['c8e60637-de79-4334-8daf-d35f18070c29', "anneu1"] ,['c660af59-5803-4846-b36e-ab61afebe081', "KS"]  ,  ['dda5fc59-f09a-4256-9fb5-66c67667a466', "anneu2"], ["a4000c2f-fa75-4b3e-8f06-a7cf599b87ad", "Karolina"]]
    #expIDs=[['c8e60637-de79-4334-8daf-d35f18070c29', "anneu1"] ,['c660af59-5803-4846-b36e-ab61afebe081', "KS"]  ,  ['dda5fc59-f09a-4256-9fb5-66c67667a466', "anneu2"], ["a4000c2f-fa75-4b3e-8f06-a7cf599b87ad", "Karolina"]]
    #expIDs=[["ecb5520d-1358-434c-95ec-93687ecd1396", "Anne"]]
    #expIDs= [['c9fec76e-7a20-4da4-93ad-04510a89473b', "nate"]]
    #expIDs=[['dda5fc59-f09a-4256-9fb5-66c67667a466', "anneu2"]]
    #expI3Ds=[["a4000c2f-fa75-4b3e-8f06-a7cf599b87ad", "Karolina"]]
    expIDs=[["ecb5520d-1358-434c-95ec-93687ecd1396", "Anne"], ['c9fec76e-7a20-4da4-93ad-04510a89473b', "nate"]]
    for k in expIDs:
        try: 

            i=k[0]
            print(k[1])
            print("probe00")
            file_p=open(k[1]+"_probe00","wb")
            table=general_analysis(i, 0, base_layout=0)
            pickle.dump(table, file_p )
            file_p.close()

            
            #general_analysis(i, 0, base_layout=0, region_list=["VIS"])
            #general_analysis(i, 0, base_layout=0, region_list=["CA1", "CA3"])
            #input("Enter your value: ") 
            #general_analysis(i, "probe00",feedbackType=1)
            #general_analysis(i, "probe00", feedbackType=0)
            #print("probe01")
            #general_analysis(i, 1,base_layout=0)
            #input("Enter your value: ") 
            #print("both")
            #general_analysis(i, "both")
            

            #input("Enter your value: ") 
        except:
            print(k[1] +" did not work")

    





        