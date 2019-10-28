from package.rivuletpy.rivuletpy.utils.io import loadswc
from package.rivuletpy.rivuletpy.utils.graph_helper import swc_to_total_node_num
from package.rivuletpy.rivuletpy.utils.graph_helper import create_swcindex_nodeid_map
from package.rivuletpy.rivuletpy.utils.graph_helper import create_tree_map
import numpy as np
import collections
import math
import sys
sys.setrecursionlimit(10000)


def get_node_id_pid(swc_array_row_number_input, swc_array_input):
    node_id = int(swc_array_input[swc_array_row_number_input, 0])
    node_pid = int(swc_array_input[swc_array_row_number_input, 6])
    return node_id, node_pid


def edge_exist(node_id_input, node_pid_input, tree_map_input):
    edge_exist_debug = False
    node_id_nonexist = True
    node_pid_nonexist = True
    edge_exist = False
    if edge_exist_debug:
        print('edge_exist is being called')
    if edge_exist_debug:
        print('node_id_input',
              node_id_input,
              'node_pid_input',
              node_pid_input)
    if node_id_input in tree_map_input.keys():
        node_id_nonexist = False
    if node_pid_input in tree_map_input.keys():
        node_pid_nonexist = False
    if node_pid_nonexist == False and node_id_nonexist == False:
        if tree_map_input[node_id_input] == node_pid_input:
            edge_exist = True
    if node_id_input == -1 or node_pid_input == -1 or node_id_nonexist or node_pid_nonexist:
        return False
    elif edge_exist:
        return True


def node_id_to_location(swc_array_input,
                        node_id_input,
                        swcindex_nodeid_map_input):
    swc_array_row_number = swcindex_nodeid_map_input[node_id_input]
    node_x = swc_array_input[swc_array_row_number, 2]
    node_y = swc_array_input[swc_array_row_number, 3]
    node_z = swc_array_input[swc_array_row_number, 4]
    return node_x, node_y, node_z


def edge_to_node(swc_array_input, node_id_input, node_pid_input, swcindex_nodeid_map_input, newnode_id_input):
    node1_x, node1_y, node1_z = node_id_to_location(swc_array_input=swc_array_input,
                                                    node_id_input=node_id_input,
                                                    swcindex_nodeid_map_input=swcindex_nodeid_map_input)
    node2_x, node2_y, node2_z = node_id_to_location(swc_array_input=swc_array_input,
                                                    node_id_input=node_pid_input,
                                                    swcindex_nodeid_map_input=swcindex_nodeid_map_input)
    new_node_x = (node1_x + node2_x) / 2
    new_node_y = (node1_y + node2_y) / 2
    new_node_z = (node1_z + node2_z) / 2
    new_node_radius = 1
    new_node_structure = 1
    new_node = np.asarray([newnode_id_input,
                           new_node_structure,
                           new_node_x,
                           new_node_y,
                           new_node_z,
                           new_node_radius,
                           node_id_input,
                           node_pid_input])
    return new_node


def unit_helper2():
    swcpath = '/home/donghao/Desktop/Liver_vessel/dh_test/runs/vessel_pred_0.1.swc'
    swc = loadswc(swcpath)
    swc_copy = swc.copy()
    total_node_number = swc_to_total_node_num(swc_array_input=swc_copy)
    tree_map = create_tree_map(swc_array_input=swc_copy,
                               total_node_number_input=total_node_number)
    swcindex_nodeid_map = create_swcindex_nodeid_map(swc_array_input=swc_copy)
    test_row_number = 3
    node_id, node_pid = get_node_id_pid(swc_array_row_number_input=test_row_number,
                                        swc_array_input=swc_copy)
    test_edge_exist = edge_exist(node_id_input=node_id,
                                 node_pid_input=node_pid,
                                 tree_map_input=tree_map)
    newnode_id = 1
    node_x, node_y, node_z = node_id_to_location(swc_array_input=swc_copy,
                                                 node_id_input=node_id,
                                                 swcindex_nodeid_map_input=swcindex_nodeid_map)
    newnode = edge_to_node(swc_array_input=swc_copy,
                           node_id_input=node_id,
                           node_pid_input=node_pid,
                           swcindex_nodeid_map_input=swcindex_nodeid_map,
                           newnode_id_input=newnode_id)
    new_node_counter = 0
    for cur_num in range(total_node_number):

        node_id, node_pid = get_node_id_pid(swc_array_row_number_input=cur_num,
                                            swc_array_input=swc_copy)
        test_edge_exist = edge_exist(node_id_input=node_id,
                                     node_pid_input=node_pid,
                                     tree_map_input=tree_map)
        if test_edge_exist:
            newnode = edge_to_node(swc_array_input=swc_copy,
                                   node_id_input=node_id,
                                   node_pid_input=node_pid,
                                   swcindex_nodeid_map_input=swcindex_nodeid_map,
                                   newnode_id_input=newnode_id)
            new_node_counter = new_node_counter + 1
            if new_node_counter == 1:
                newnode_all = newnode
            else:
                newnode_all = np.vstack((newnode_all, newnode))
    #     print('cur_num',
    #           cur_num,
    #           'new_node_counter',
    #           new_node_counter,
    #           'node_id',
    #           node_id,
    #           'node_pid',
    #           node_pid,
    #           'test_edge_exist',
    #           test_edge_exist,
    #           'newnode',
    #           newnode)
    print('Total number of new nodes is', new_node_counter)
    print('The shape of newnode_all', newnode_all.shape)
    print('The total_node_number is', total_node_number)