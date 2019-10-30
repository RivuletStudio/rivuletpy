from package.rivuletpy.rivuletpy.utils.io import loadswc
from package.rivuletpy.rivuletpy.utils.graph_helper import swc_to_total_node_num
from package.rivuletpy.rivuletpy.utils.graph_helper import create_swcindex_nodeid_map
from package.rivuletpy.rivuletpy.utils.graph_helper import create_tree_map
from package.rivuletpy.rivuletpy.utils.graph_helper import swc_to_branch_id
from package.rivuletpy.rivuletpy.utils.graph_helper2 import get_node_id_pid
from package.rivuletpy.rivuletpy.utils.graph_helper2 import edge_exist
from package.rivuletpy.rivuletpy.utils.graph_helper2 import edge_to_node

import numpy as np
import collections
import math
import sys
sys.setrecursionlimit(10000)


def find_node_pid(swc_array_input,
                  node_id_input,
                  swcindex_nodeid_map_input):
    swc_array_row_number = swcindex_nodeid_map_input[node_id_input]
    node_pid = swc_array_input[swc_array_row_number, 6]
    return node_pid


def reverse_tree_map(tree_map_input):
    reversed_tree_map = {}
    for cur_node in tree_map_input:
        cur_node_pid = tree_map_input[cur_node]
        if cur_node_pid in list(reversed_tree_map.keys()):
            child_list = reversed_tree_map[cur_node_pid]
            child_list = np.hstack((child_list, np.asarray(cur_node)))
            reversed_tree_map[cur_node_pid] = child_list
        else:
            reversed_tree_map[cur_node_pid] = np.asarray(cur_node)
    return reversed_tree_map


def create_all_newnode(swc_array_input, tree_map_input, swcindex_nodeid_map_input, total_node_number_input):
    new_node_counter = 0
    for cur_num in range(total_node_number_input):
        node_id, node_pid = get_node_id_pid(swc_array_row_number_input=cur_num,
                                            swc_array_input=swc_array_input)
        test_edge_exist = edge_exist(node_id_input=node_id,
                                     node_pid_input=node_pid,
                                     tree_map_input=tree_map_input)
        if test_edge_exist:
            newnode = edge_to_node(swc_array_input=swc_array_input,
                                   node_id_input=node_id,
                                   node_pid_input=node_pid,
                                   swcindex_nodeid_map_input=swcindex_nodeid_map_input,
                                   newnode_id_input=new_node_counter)
            new_node_counter = new_node_counter + 1
            if new_node_counter == 1:
                newnode_all = newnode
            else:
                newnode_all = np.vstack((newnode_all, newnode))
    return newnode_all


def find_node_children(node_id_input, reversed_tree_map_input):
    node_children = reversed_tree_map_input[node_id_input]
    return node_children


def node_to_edge(swc_array_input,
                 node_id_input,
                 swcindex_nodeid_map_input,
                 reversed_tree_map_input,
                 newnode_all_input):
    node_pid = find_node_pid(swc_array_input=swc_array_input,
                             node_id_input=node_id_input,
                             swcindex_nodeid_map_input=swcindex_nodeid_map_input)
    if node_id_input in reversed_tree_map_input.keys():
        node_children = find_node_children(node_id_input=node_id_input,
                                           reversed_tree_map_input=reversed_tree_map_input)
    else:
        return np.asarray([-6, -6, -6, -6])
    connection_set = node_children.copy()
    connection_set = np.append(connection_set,node_pid)
    connection_set = connection_set.astype('int')
    collection_set = connection_set[connection_set!=-1]
    connection_num = collection_set.size
    if connection_num == 1:
        edge_vec = np.asarray([-6, -6, -6, -6])
    elif connection_num == 2:
        edge_vec = np.asarray([node_id_input, collection_set[0], collection_set[1], node_id_input])
    else:
        for connection_i in range(len(connection_set)):
            for connection_node2 in connection_set[connection_i+1:]:
                if connection_i == 0:
                    edge_vec = np.asarray([node_id_input,connection_set[connection_i],connection_node2, node_id_input])
                else:
                    edge_vec = np.vstack((edge_vec, [node_id_input,connection_set[connection_i],connection_node2, node_id_input]))
    return edge_vec


def find_pairs_from_pairset(id1, id2, newnode_all_input):
    if id1 > id2:
        target_small_id = id2
        target_large_id = id1
    else:
        target_small_id = id1
        target_large_id = id2
    total_node_num = newnode_all_input.shape[0]
    for node_i in range(0, total_node_num, 1):
        source_id1 = int(newnode_all_input[node_i, 6])
        source_id2 = int(newnode_all_input[node_i, 7])
        if source_id1 == target_small_id and source_id2 == target_large_id:
            return node_i
        if source_id1 == target_large_id and source_id2 == target_small_id:
            return node_i
    return -6


def edge_vec_to_edge_nodes(edge_vec_input, newnode_all_input):
    if edge_vec_input.size==4:
        edge_node1 = find_pairs_from_pairset(id1=edge_vec_input[0],
                                             id2=edge_vec_input[1],
                                             newnode_all_input=newnode_all_input)
        edge_node2 = find_pairs_from_pairset(id1=edge_vec_input[2],
                                             id2=edge_vec_input[3],
                                             newnode_all_input=newnode_all_input)
        edge_list = np.asarray([edge_node1, edge_node2])
    elif edge_vec_input.size==8:
        edge_node1 = find_pairs_from_pairset(id1=edge_vec_input[0,0],
                                             id2=edge_vec_input[0,1],
                                             newnode_all_input=newnode_all_input)
        edge_node2 = find_pairs_from_pairset(id1=edge_vec_input[0,2],
                                             id2=edge_vec_input[0,3],
                                             newnode_all_input=newnode_all_input)
        edge_list = np.asarray([edge_node1, edge_node2])
        edge_node3 = find_pairs_from_pairset(id1=edge_vec_input[1,0],
                                             id2=edge_vec_input[1,1],
                                             newnode_all_input=newnode_all_input)
        edge_list = np.vstack((edge_list, edge_node3))
        edge_node4 = find_pairs_from_pairset(id1=edge_vec_input[1,2],
                                             id2=edge_vec_input[1,3],
                                             newnode_all_input=newnode_all_input)
        edge_list = np.vstack((edge_list, edge_node4))
    else:
        edge_node1 = find_pairs_from_pairset(id1=edge_vec_input[0,0],
                                             id2=edge_vec_input[0,1],
                                             newnode_all_input=newnode_all_input)
        edge_node2 = find_pairs_from_pairset(id1=edge_vec_input[0,2],
                                             id2=edge_vec_input[0,3],
                                             newnode_all_input=newnode_all_input)
        edge_list = np.asarray([edge_node1, edge_node2])
        edge_num = edge_vec_input.shape[0]
        for edge_i in range(1, edge_num, 1):
            cur_edge_node1 = find_pairs_from_pairset(id1=edge_vec_input[edge_i,0],
                                                     id2=edge_vec_input[edge_i,1],
                                                     newnode_all_input=newnode_all_input)
            cur_edge_node2 = find_pairs_from_pairset(id1=edge_vec_input[edge_i,2],
                                                     id2=edge_vec_input[edge_i,3],
                                                     newnode_all_input=newnode_all_input)
            edge_list = np.vstack((edge_list, [cur_edge_node1, cur_edge_node2]))
    return edge_list


def unit_helper3():
    swcpath = '/home/donghao/Desktop/Liver_vessel/dh_test/runs/vessel_pred_0.1.swc'
    swc = loadswc(swcpath)
    swc_copy = swc.copy()
    total_node_number = swc_to_total_node_num(swc_array_input=swc_copy)
    tree_map = create_tree_map(swc_array_input=swc_copy,
                               total_node_number_input=total_node_number)
    swcindex_nodeid_map = create_swcindex_nodeid_map(swc_array_input=swc_copy)
    reversed_tree_map = reverse_tree_map(tree_map_input=tree_map)
    newnode_all = create_all_newnode(swc_array_input=swc_copy,
                                     tree_map_input=tree_map,
                                     swcindex_nodeid_map_input=swcindex_nodeid_map,
                                     total_node_number_input=total_node_number)
    cur_node_counter = 0
    for cur_node in tree_map.keys():
        edge_vec = node_to_edge(swc_array_input=swc_copy,
                                node_id_input=cur_node,
                                swcindex_nodeid_map_input=swcindex_nodeid_map,
                                reversed_tree_map_input=reversed_tree_map,
                                newnode_all_input=newnode_all)
        if cur_node_counter == 0:
            edge_vec_list = edge_vec
        else:
            edge_vec_list = np.vstack((edge_vec_list, edge_vec))
        cur_node_counter = cur_node_counter + 1
    edge_list = edge_vec_to_edge_nodes(edge_vec_input=edge_vec_list, newnode_all_input=newnode_all)
    print(edge_list)

    print('-----UNIT TEST BEGINS-----')
    swcpath = '/home/donghao/Desktop/Liver_vessel/dh_test/runs/vessel_pred_0.1.swc'
    swc = loadswc(swcpath)
    swc_copy = swc.copy()
    total_node_number = swc_to_total_node_num(swc_array_input=swc_copy)
    tree_map = create_tree_map(swc_array_input=swc_copy,
                               total_node_number_input=total_node_number)
    swcindex_nodeid_map = create_swcindex_nodeid_map(swc_array_input=swc_copy)
    reversed_tree_map = reverse_tree_map(tree_map_input=tree_map)
    node_id = 98
    node_pid = find_node_pid(swc_array_input=swc_copy,
                             node_id_input=node_id,
                             swcindex_nodeid_map_input=swcindex_nodeid_map)
    node_children = find_node_children(node_id_input=node_id,
                                       reversed_tree_map_input=reversed_tree_map)
    print('node_id', node_id, 'node_children', node_children)
    newnode_all = create_all_newnode(swc_array_input=swc_copy,
                                     tree_map_input=tree_map,
                                     swcindex_nodeid_map_input=swcindex_nodeid_map,
                                     total_node_number_input=total_node_number)
    edge_vec = node_to_edge(swc_array_input=swc_copy,
                            node_id_input=node_id,
                            swcindex_nodeid_map_input=swcindex_nodeid_map,
                            reversed_tree_map_input=reversed_tree_map,
                            newnode_all_input=newnode_all)
    print('edge_vec', edge_vec)
    # newnode_id_input(0), new_node_structure(1),
    # new_node_x(2), new_node_y(3), new_node_z(4),
    # new_node_radius(5), node_id_input(6), node_pid_input(7)
    test_node_i = find_pairs_from_pairset(id1=170, id2=171, newnode_all_input=newnode_all)
    edgelist = edge_vec_to_edge_nodes(edge_vec_input=edge_vec, newnode_all_input=newnode_all)
    print('edgelist', edgelist)
    parent_id_branch, parent_id_branch_times = swc_to_branch_id(swc_array_input=swc_copy)
    print('the shape of parent_id_branch', parent_id_branch.shape)
    parent_id_branch = np.squeeze(parent_id_branch)
    parent_id_branch_times = np.squeeze(parent_id_branch_times)
    print('parent_id_branch',
          parent_id_branch,
          'parent_id_branch_times',
          parent_id_branch_times)
    print('-----UNIT TEST ENDS-----')