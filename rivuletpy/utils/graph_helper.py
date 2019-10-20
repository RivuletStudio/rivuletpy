from package.rivuletpy.rivuletpy.utils.io import loadswc
from package.rivuletpy.rivuletpy.utils.metrics import upsample_swc
import numpy as np
import collections
import math
import sys
sys.setrecursionlimit(10000)


def create_swcindex_nodeid_map(swc_array_input):
    swcindex_nodeid_map = {}
    # Build a nodeid->idx hash table
    for swcindex_i in range(swc_array_input.shape[0]):
        swcindex_nodeid_map[swc_array_input[swcindex_i, 0]] = swcindex_i
    return swcindex_nodeid_map


def swc_to_total_node_num(swc_array_input):
    total_node_number = swc_array_input.shape[0]
    debug = False
    if debug:
        print('total_node_number',
              total_node_number)
    return total_node_number


def create_tree_map(swc_array_input, total_node_number_input):
    tree_map = {}
    debug = False
    for node_i in range(total_node_number_input):
        node_id = swc_array_input[node_i, 0]
        node_pid = swc_array_input[node_i, 6]
        tree_map[int(node_id)] = int(node_pid)
    if debug:
        print('the length of tree map', len(tree_map))
    return tree_map


def swc_to_branch_id(swc_array_input):
    parent_ids, parent_id_times = np.unique(swc_array_input[:, 6],
                                            return_counts=True)
    parent_id_branch_index = np.argwhere(parent_id_times > 1)
    parent_id_branch = parent_ids[parent_id_branch_index]
    parent_id_branch = parent_id_branch.astype('int')
    print('the shape of parent_id_branch', parent_id_branch.shape)
    return parent_id_branch


def downsample_swc(child_node_input,
                   swc_array_input,
                   swcindex_nodeid_map_input,
                   sampling_length_input,
                   tree_map_copy_input,
                   current_path_input,
                   node_x_path_input,
                   node_y_path_input,
                   node_z_path_input,
                   current_length_input,
                   sampled_tree_map_input):
    if child_node_input in tree_map_copy_input.keys():
        swc_array_row_number = swcindex_nodeid_map_input[child_node_input]
    else:
        swc_array_row_number = list(tree_map_copy_input.keys())[0]
    node_id = int(swc_array_input[swc_array_row_number, 0])
    node_structure = swc_array_input[swc_array_row_number, 1]
    node_x = swc_array_input[swc_array_row_number, 2]
    node_y = swc_array_input[swc_array_row_number, 3]
    node_z = swc_array_input[swc_array_row_number, 4]
    node_radius = swc_array_input[swc_array_row_number, 5]
    node_pid = int(swc_array_input[swc_array_row_number, 6])
    debug = False

    if debug:
        print('node_id',
              node_id,
              'node_structure',
              node_structure,
              'x y z',
              node_x,
              node_y,
              node_z,
              'radius',
              node_radius,
              'node_pid',
              node_pid)
    current_path_input.append(node_id)
    node_x_path_input.append(node_x)
    node_y_path_input.append(node_y)
    node_z_path_input.append(node_z)

    if debug:
        print('current_path_input',
              current_path_input,
              'node_x_path_input',
              node_x_path_input,
              'node_y_path_input',
              node_y_path_input,
              'node_z_path_input',
              node_z_path_input)
    current_length = 0
    node_path_length = len(current_path_input)

    if debug:
        print('node_path_length',
              node_path_length,
              'current_length_input',
              current_length_input)

    if len(current_path_input) > 1.5:
        for i in range(1, len(current_path_input)):
            current_length = (node_x_path_input[i] - node_x_path_input[i - 1]) * (
                        node_x_path_input[i] - node_x_path_input[i - 1]) + current_length
            current_length = (node_y_path_input[i] - node_y_path_input[i - 1]) * (
                        node_y_path_input[i] - node_y_path_input[i - 1]) + current_length
            current_length = (node_z_path_input[i] - node_z_path_input[i - 1]) * (
                        node_z_path_input[i] - node_z_path_input[i - 1]) + current_length
    current_length_input = current_length_input + math.sqrt(current_length)

    if node_path_length > 2 and current_length_input > sampling_length_input:
        for remove_i in range(0, node_path_length, 1):
            cur_node_id = current_path_input[remove_i]
            if debug:
                print('cur_node_id', cur_node_id)
            if remove_i == 0:
                if debug:
                    print('if condition remove_i', remove_i)
                start_node_id = cur_node_id
                end_node_id = sampled_tree_map_input[start_node_id]
            elif remove_i == node_path_length - 1:
                if debug:
                    print('elif condition remove_i', remove_i)
                    print('end_node_id', end_node_id)
                if end_node_id != -1:
                    end_node_id_parent = sampled_tree_map_input[cur_node_id]
                    sampled_tree_map_input.pop(cur_node_id)
                    sampled_tree_map_input[end_node_id] = end_node_id_parent
            else:
                if debug:
                    print('else condition remove_i', remove_i)
                if cur_node_id in sampled_tree_map_input.keys():
                    sampled_tree_map_input.pop(cur_node_id)
                    if debug:
                        print('the length of sampled_tree_map_input', len(sampled_tree_map_input))

    if current_length_input > sampling_length_input:
        current_path_input = []
        node_x_path_input = []
        node_y_path_input = []
        node_z_path_input = []
        current_length_input = 0

    if node_id in tree_map_copy_input.keys():
        tree_map_copy_input.pop(node_id)
    else:
        print('node_id is not found.')

    if len(tree_map_copy_input) > 1:
        return downsample_swc(child_node_input=node_pid,
                              swc_array_input=swc_array_input,
                              swcindex_nodeid_map_input=swcindex_nodeid_map_input,
                              sampling_length_input=sampling_length_input,
                              tree_map_copy_input=tree_map_copy_input,
                              current_path_input=current_path_input,
                              node_x_path_input=node_x_path_input,
                              node_y_path_input=node_y_path_input,
                              node_z_path_input=node_z_path_input,
                              current_length_input=current_length_input,
                              sampled_tree_map_input=sampled_tree_map_input)
    else:
        return sampled_tree_map_input


swcpath = '/home/donghao/Desktop/Liver_vessel/dh_test/runs/vessel_pred_0.1.swc'
swc = loadswc(swcpath)
swc_copy = swc.copy()
swcindex_nodeid_map = create_swcindex_nodeid_map(swc_array_input=swc_copy)
total_node_number = swc_to_total_node_num(swc_array_input=swc_copy)
tree_map = create_tree_map(swc_array_input=swc_copy,
                           total_node_number_input=total_node_number)
tree_map_copy = tree_map.copy()
sampled_tree_map = tree_map.copy()
current_path = []
node_x_path = []
node_y_path = []
node_z_path = []
ini_length = 0
last_node_id = swc_copy[0, 0]
sampling_length = 3
output_downsample_swc = downsample_swc(child_node_input=last_node_id,
                                       swc_array_input=swc_copy,
                                       swcindex_nodeid_map_input=swcindex_nodeid_map,
                                       sampling_length_input=sampling_length,
                                       tree_map_copy_input=tree_map_copy,
                                       current_path_input=current_path,
                                       node_x_path_input=node_x_path,
                                       node_y_path_input=node_y_path,
                                       node_z_path_input=node_z_path,
                                       current_length_input=ini_length,
                                       sampled_tree_map_input=sampled_tree_map)
# print('output_downsample_swc', output_downsample_swc)
print('the length of tree_map before downsampling', len(tree_map))
print('the length of tree_map after downsampling', len(output_downsample_swc))