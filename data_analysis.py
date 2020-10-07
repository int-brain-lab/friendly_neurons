from friendly_neurons import *
import matplotlib.pyplot as plt
import random
import igraph

from oneibl.one import ONE





def general_analysis(exp_ID,probe, visual=False, height=1000, length=1000,feedbackType=None, base_layout=1):
    
    pre_graph,pre_partition, pre_regions, locations=brain_region(exp_ID, visual=visual, probe=probe, user_start="trial_start", user_end="stimOn_times", feedbackType=feedbackType)
    graph,partition, regions, locations=brain_region(exp_ID, visual=visual, probe=probe,user_start="stimOn_times",user_end="response_times",feedbackType= feedbackType)
    post_graph, post_partition, pre_regions, locations =brain_region(exp_ID, visual=visual, probe=probe, user_start="response_times", user_end="trial_end", feedbackType=feedbackType)
    print("StimOn")
    print(pre_graph.summary(verbosity=1))
    print("Response")
    print(graph.summary(verbosity=1))
    print("End")
    print(post_graph.summary(verbosity=1))
    graphs_for_partitions=[pre_graph, graph, post_graph]
    print("Split join distance")
    print([[ split_join_distance(graphs_for_partitions[i],graphs_for_partitions[j] ) for i in range(len(graphs_for_partitions))] for j in range(len(graphs_for_partitions))])
    print("Compare communities")
    print([[ compare_communities(graphs_for_partitions[i],graphs_for_partitions[j] ) for i in range(len(graphs_for_partitions))] for j in range(len(graphs_for_partitions))])
    print("Ovelap of communities")
    print(partition)
    print(com_overlap(partition, [ pre_partition, post_partition]))
    print(len(pre_graph.graph.vs["label"]))

    print("locations"+str(len(locations)))
    ##Locations by community
    locations_simplified=[parse(i) for i in locations]
    print("locations"+str(len(locations_simplified)))
    
    #visualize(pre_graph, layout=locations_from_dictionary(pre_partition,length=length, height=height), vertex_size=30, labels=locations_simplified, length=length, height=height)
    locations_to_order=[ i for i in set(locations_simplified)]
    locations_to_order.sort()
    print(locations_to_order)
    order_x=labels_to_dictionary(locations_to_order)
    for i in order_x:
        temp=0
        for k in order_x[i]:
            temp=k
        order_x[i]=temp
    ##Locations by brain region 
    


    
    ## locations with brain region on Y axis and cluster on X axis 
    #layout_x=locations_from_dictionary(pre_partition,length=length, height=height, seperated=False)
    if base_layout==0:
        layout_x=locations_from_dictionary(pre_partition,length=length, height=height, seperated=False)
    else:
        layout_x=locations_from_dictionary(partition,length=length, height=height, seperated=False)
    layout_y=locations_from_dictionary(labels_to_dictionary(locations_simplified),order=order_x , length=length, height=height)
    layout_mixed=[ [layout_x[i][0],layout_y[i][1]] for i in range(min(len(layout_y),len(layout_x)))]
    coloring= ClusterColoringPalette(10)
    visualize(pre_graph, layout= layout_mixed, vertex_size=40, labels=locations_simplified, length=length, height=height, coloring=coloring)
    visualize(graph, layout= layout_mixed, vertex_size=40, labels=locations_simplified, length=length, height=height,coloring=coloring)
    visualize(post_graph, layout= layout_mixed, vertex_size=40, labels=locations_simplified, length=length, height=height ,coloring=coloring)
    

    
    
    belong_pre=belong_dictionary([pre_partition])
    print(belong_pre)
    #dense=belong_dictionary(partition)
    #print(dense)
    #dense_post=belong_dictionary(post_partition)


    print(pre_regions)
    print(sections(pre_regions))
    print(locations)

class Pallete_changed(ClusterColoringPalette):

  

    pass

def match_colors():
    pass


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
    


def locations_organized(): 
    pass



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

def belong_dictionary(partitions):

    result=dict()
    for i in range( len(partitions)): 
        for j in partitions[i]: 
            for k in partitions[i][j]:
                if k in result:
                    result[k].append((i,j))
                else:
                    result[k]=[(i,j)]
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
    plot(g, **visual_style1)



def graph_from_dict():
    pass




#expIDs=[["ecb5520d-1358-434c-95ec-93687ecd1396", "Anne"],['c8e60637-de79-4334-8daf-d35f18070c29', "anneu1"] ,['c660af59-5803-4846-b36e-ab61afebe081', "KS"] ,['c9fec76e-7a20-4da4-93ad-04510a89473b', "nate"] ,  ['dda5fc59-f09a-4256-9fb5-66c67667a466', "anneu2"], ["a4000c2f-fa75-4b3e-8f06-a7cf599b87ad", "Karolina"]]
expIDs=[["ecb5520d-1358-434c-95ec-93687ecd1396", "Anne"]]
one=ONE()


for k in expIDs:
    i=k[0]
    print(k[1])
    print("probe00")
    general_analysis(i, "probe00")

    input("Enter your value: ") 
    #general_analysis(i, "probe00",feedbackType=1)
    #general_analysis(i, "probe00", feedbackType=0)
    print("probe01")
    general_analysis(i, "probe01",base_layout=0)
    input("Enter your value: ") 
    general_analysis(i, "both")
    print("both")
    #general_analysis(i, "both")
    input("Enter your value: ") 

    





        