import os
import math
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.spatial.distance import cdist
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from ..utils.io import loadswc
from patch_extract import load_ct_im_dh


def get_subs_from_test_list(config_input):
    list_txt = config_input.test_list_txt
    if not os.path.isfile(list_txt):
        raise ValueError('the list file does not exist!')
    subs = [sub.rstrip('\n').rstrip('\r') for sub in open(list_txt, 'r').readlines()]
    return subs


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


def create_swcindex_nodeid_map(swc_array_input):
    swcindex_nodeid_map = {}
    # Build a nodeid->idx hash table
    for swcindex_i in range(swc_array_input.shape[0]):
        swcindex_nodeid_map[swc_array_input[swcindex_i, 0]] = swcindex_i
    return swcindex_nodeid_map


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


def get_newnodes_from_swc(swc_array_input):
    total_node_number_f = swc_to_total_node_num(swc_array_input=swc_array_input)
    tree_map_f = create_tree_map(swc_array_input=swc_array_input,
                                 total_node_number_input=total_node_number_f)
    swcindex_nodeid_map_f = create_swcindex_nodeid_map(swc_array_input=swc_array_input)
    newnodes_output = create_all_newnode(swc_array_input=swc_array_input,
                                         tree_map_input=tree_map_f,
                                         swcindex_nodeid_map_input=swcindex_nodeid_map_f,
                                         total_node_number_input=total_node_number_f)
    return newnodes_output


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


def find_node_pid(swc_array_input,
                  node_id_input,
                  swcindex_nodeid_map_input):
    swc_array_row_number = swcindex_nodeid_map_input[node_id_input]
    node_pid = swc_array_input[swc_array_row_number, 6]
    return node_pid


def find_node_children(node_id_input, reversed_tree_map_input):
    node_children = reversed_tree_map_input[node_id_input]
    return node_children


def node_to_edge(swc_array_input,
                 node_id_input,
                 swcindex_nodeid_map_input,
                 reversed_tree_map_input):
    node_pid = find_node_pid(swc_array_input=swc_array_input,
                             node_id_input=node_id_input,
                             swcindex_nodeid_map_input=swcindex_nodeid_map_input)
    if node_id_input in reversed_tree_map_input.keys():
        node_children = find_node_children(node_id_input=node_id_input,
                                           reversed_tree_map_input=reversed_tree_map_input)
    else:
        return np.asarray([-6, -6, -6, -6])
    connection_set = node_children.copy()
    connection_set = np.append(connection_set, node_pid)
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
                    edge_vec = np.asarray([node_id_input, connection_set[connection_i], connection_node2, node_id_input])
                else:
                    edge_vec = np.vstack((edge_vec, [node_id_input, connection_set[connection_i], connection_node2, node_id_input]))
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


def create_edge_list(swc_array_input,
                     newnode_all_input):
    total_node_number_f = swc_to_total_node_num(swc_array_input=swc_array_input)
    tree_map_f = create_tree_map(swc_array_input=swc_array_input,
                                 total_node_number_input=total_node_number_f)
    swcindex_nodeid_map_f = create_swcindex_nodeid_map(swc_array_input=swc_array_input)
    reversed_tree_map_f = reverse_tree_map(tree_map_input=tree_map_f)
    cur_node_counter = 0
    for cur_node in tree_map_f.keys():
        edge_vec = node_to_edge(swc_array_input=swc_array_input,
                                node_id_input=cur_node,
                                swcindex_nodeid_map_input=swcindex_nodeid_map_f,
                                reversed_tree_map_input=reversed_tree_map_f)
        if cur_node_counter == 0:
            edge_vec_list = edge_vec
        else:
            edge_vec_list = np.vstack((edge_vec_list, edge_vec))
        cur_node_counter = cur_node_counter + 1
    edge_list = edge_vec_to_edge_nodes(edge_vec_input=edge_vec_list,
                                       newnode_all_input=newnode_all_input)
    return edge_list


def post_processing_edge_list(edge_list_input):
    save_index_list = []
    for edge_i, cur_edge in enumerate(edge_list_input):
        if cur_edge[0] != -6 and cur_edge[1] != -6:
            save_index_list.append(edge_i)
    new_edge_list = edge_list_input[ save_index_list, :]
    return new_edge_list


def vtk2swc(vtk_file_name):
    reader = vtk.vtkDataSetReader()
    reader.SetFileName(vtk_file_name)
    reader.ReadAllScalarsOn()  # Activate the reading of all scalars
    reader.Update()
    data = reader.GetOutput()
    tree1_points = vtk_to_numpy(data.GetPoints().GetData())
    tree1_lines = vtk_to_numpy(data.GetLines().GetData())
    tree1_lines_array = tree1_lines.reshape((-1, 3))
    if tree1_lines_array.shape[0] != tree1_points.shape[0]:
        # TODO Mismatch of points and lines. Please fix it later.
        # Create swc as N x 7 numpy array
        swc = np.zeros([tree1_lines_array.shape[0], 7])
        for cur_id, cur_line in enumerate(tree1_lines_array):
            # sample from vtk
            print('cur_line', cur_line)
            swc[cur_id, 0] = tree1_lines_array[cur_id, 1]
            # parent id from vtk
            swc[cur_id, 6] = tree1_lines_array[cur_id, 2]
            # x, y, z
            swc[cur_id, 2] = (-1) * tree1_points[tree1_lines_array[cur_id, 1], 0]
            swc[cur_id, 3] = (-1) * tree1_points[tree1_lines_array[cur_id, 1], 1]
            swc[cur_id, 4] = tree1_points[tree1_lines_array[cur_id, 1], 2]
            # Set all type to axon
            swc[cur_id, 1] = 2
    else:
        # Create swc as N x 7 numpy array
        swc = np.zeros([tree1_points.shape[0], 7])
        swc[:, 2] = tree1_points[:, 0]
        swc[:, 3] = tree1_points[:, 1]
        swc[:, 4] = tree1_points[:, 2]
        # Set the radius of array to 1
        swc[:, 5] = 1
        # Set all type to axon
        swc[:, 1] = 2
        # sample from vtk
        swc[:, 0] = tree1_lines_array[:, 1]
        # parent id from vtk
        swc[:, 6] = tree1_lines_array[:, 2]
    return swc


def get_gtswc_from_sub(sub_input):
    save_tracing_path = './dh_test_bk_pts_online/tracing_eval'
    gt_vtk_tree_path = os.path.join(save_tracing_path,
                                    'pt_vessel_gt',
                                    sub_input)
    tree_names = os.listdir(gt_vtk_tree_path)
    tree_counter = 0
    for tree in tree_names:
        if tree.endswith(".vtk"):
            print('tree', tree)
            if tree_counter == 0:
                tree1_path = gt_vtk_tree_path + '/' + tree
                gt_swc = vtk2swc(tree1_path)
            if tree_counter > 0:
                tree_path = gt_vtk_tree_path + '/' + tree
                gt2_swc = vtk2swc(tree_path)
                rand_i = np.random.randint(low=2, high=30)
                gt2_swc[:, 0] = gt2_swc[:, 0] + 80000 * rand_i
                gt2_swc[:, 6] = gt2_swc[:, 6] + 80000 * rand_i
                gt_swc = np.concatenate((gt_swc, gt2_swc), axis=0)
            tree_counter = tree_counter + 1
    return gt_swc


def matched_pt(pt_input, gt_swc_input):
    d = cdist(pt_input, gt_swc_input[:, 2:5])
    mindist1 = d.min(axis=1)
    mindist1_index = np.argmin(d)
    match_node_x1_output = gt_swc_input[mindist1_index, 2]
    match_node_y1_output = gt_swc_input[mindist1_index, 3]
    match_node_z1_output = gt_swc_input[mindist1_index, 4]
    return mindist1, mindist1_index, match_node_x1_output, match_node_y1_output, match_node_z1_output


def edge_match_v1(pred_swc_input, cur_pred_newnode_input, pred_swcind_nodeid_map_input, gt_swc_input):
    pred_node_x1, pred_node_y1, pred_node_z1 = node_id_to_location(swc_array_input=pred_swc_input,
                                                                   node_id_input=cur_pred_newnode_input[6],
                                                                   swcindex_nodeid_map_input=pred_swcind_nodeid_map_input)
    d = cdist(np.asarray([[pred_node_x1, pred_node_y1, pred_node_z1]]), gt_swc_input[:, 2:5])
    mindist1 = d.min(axis=1)
    mindist1_index = np.argmin(d)
    # match_node_x1 = gt_swc[mindist1_index, 2]
    # match_node_y1 = gt_swc[mindist1_index, 3]
    # match_node_z1 = gt_swc[mindist1_index, 4]

    pred_node_x2, pred_node_y2, pred_node_z2 = node_id_to_location(swc_array_input=pred_swc_input,
                                                                   node_id_input=cur_pred_newnode_input[7],
                                                                   swcindex_nodeid_map_input=pred_swcind_nodeid_map_input)
    pt2_input = np.asarray([[pred_node_x2, pred_node_y2, pred_node_z1]])
    mindist2, mindist2_index, mnode_x2, mnode_y2, mnode_z2 = matched_pt(pt_input=pt2_input, gt_swc_input=gt_swc_input)
    # print('pred_node_x1, pred_node_y1, pred_node_z1', pred_node_x1, pred_node_y1, pred_node_z1)
    # print('match_node_x1 match_node_y1 match_node_z1', match_node_x1, match_node_y1, match_node_z1)
    # print('pred_node_x2, pred_node_y2, pred_node_z2', pred_node_x2, pred_node_y2, pred_node_z2)
    # print('mnode_x2, mnode_y2, mnode_z2', mnode_x2, mnode_y2, mnode_z2)
    if (abs(mindist1) + abs(mindist2)) < 10:
        return 1
    else:
        return 0


def get_newnodes_label(sub_input):
    debug_flag = False
    pred_swcpath = './dh_test_bk_pts_online/tracing_eval/pt_vessel_pred/fe131b5c-d32a-4a58-9214-b327021f63a8/vessel_pred_0.1.swc'
    pred_swc = loadswc(pred_swcpath)
    pred_swc_copy = pred_swc.copy()
    gt_swc = get_gtswc_from_sub(sub_input=sub_input)
    pred_total_node_number = swc_to_total_node_num(swc_array_input=pred_swc_copy)
    pred_tree_map = create_tree_map(swc_array_input=pred_swc_copy, total_node_number_input=pred_total_node_number)
    prediction_newnodes = get_newnodes_from_swc(swc_array_input=pred_swc_copy)
    pred_swcindex_nodeid_map = create_swcindex_nodeid_map(swc_array_input=pred_swc_copy)
    match_flag_vec = np.zeros(prediction_newnodes.shape[0])
    pred_node_counter = 0
    for cur_pred_newnode in prediction_newnodes:
        match_flag = edge_match_v1(pred_swc_input=pred_swc_copy,
                                   cur_pred_newnode_input=cur_pred_newnode,
                                   pred_swcind_nodeid_map_input=pred_swcindex_nodeid_map,
                                   gt_swc_input=gt_swc)
        match_flag_vec[pred_node_counter] = match_flag
        if cur_pred_newnode[6] not in pred_tree_map.keys() or cur_pred_newnode[7] not in pred_tree_map.keys():
            print('pred_newnode out of tree_map dictionary', cur_pred_newnode)
        pred_node_counter = pred_node_counter + 1
    if debug_flag:
        print('match_flag_vec.shape', match_flag_vec.shape)
        print('pred_total_node_number', pred_total_node_number)
    return match_flag_vec


def get_heatmap_from_casename(sub_input, config_input):
    thre = 0.1
    liver_vessel_pred_file = os.path.join(config_input.test_data_path,
                                          'derived',
                                          'heatmap_pred',
                                          sub_input,
                                          'vessel_pred.mhd')
    ct_mhd, ct_scan = load_ct_im_dh(imfilepath=liver_vessel_pred_file,
                                    resample_flag=False)
    ct_scan = np.transpose(ct_scan, (2, 1, 0))
    ct_scan_bi = ct_scan > thre
    ct_scan_bi_copy = ct_scan_bi.copy()
    return ct_scan, ct_scan_bi_copy


def get_node_feature_from_heatmap(newnode_id_input, newnode_all_input, heatmap_input):
    cur_node = newnode_all_input[newnode_id_input, :]
    node_x = int(cur_node[2])
    node_y = int(cur_node[3])
    node_z = int(cur_node[4])
    feature_vec = heatmap_input[node_x-1:node_x+2, node_y-1:node_y+2, node_z-1:node_z+2]
    feature_vec = feature_vec.flatten()
    return feature_vec


def init_tree(swc_array_input):
    swcindex_nodeid_map_f = create_swcindex_nodeid_map(swc_array_input=swc_array_input)
    total_node_number_f = swc_to_total_node_num(swc_array_input=swc_array_input)
    tree_map_f = create_tree_map(swc_array_input=swc_array_input,
                                 total_node_number_input=total_node_number_f)
    tree_structure = {
        'swc_array': swc_array_input,
        'total_node_number': total_node_number_f,
        'tree_map': tree_map_f,
        'swcindex_nodeid_map': swcindex_nodeid_map_f
    }
    return tree_structure


def get_sampled_edge_path(sampling_length_input,
                          swc_array_input,
                          swcindex_nodeid_map_input,
                          child_node_input,
                          current_path_input,
                          node_x_path_input,
                          node_y_path_input,
                          node_z_path_input,
                          current_length_input,
                          edge_path_list_input,
                          tree_map_copy_input,
                          sampled_tree_map_input):
    debug_downsample_swc = False
    if child_node_input in tree_map_copy_input.keys():
        swc_array_row_number = swcindex_nodeid_map_input[child_node_input]
    else:
        if debug_downsample_swc:
            print('else is being called')
        print('child_node_input', child_node_input)
        child_node_input = list(tree_map_copy_input.keys())[0]
        swc_array_row_number = swcindex_nodeid_map_input[child_node_input]
    if debug_downsample_swc:
        print('swc_array_row_number', swc_array_row_number)
    node_id = int(swc_array_input[swc_array_row_number, 0])
    # node_structure = swc_array_input[swc_array_row_number, 1]
    node_x = swc_array_input[swc_array_row_number, 2]
    node_y = swc_array_input[swc_array_row_number, 3]
    node_z = swc_array_input[swc_array_row_number, 4]
    # node_radius = swc_array_input[swc_array_row_number, 5]
    node_pid = int(swc_array_input[swc_array_row_number, 6])

    current_path_input.append(node_id)
    node_x_path_input.append(node_x)
    node_y_path_input.append(node_y)
    node_z_path_input.append(node_z)

    current_length = 0
    node_path_length = len(current_path_input)

    if debug_downsample_swc:
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
    if current_length_input > 200:
        current_path_input = []
        node_x_path_input = []
        node_y_path_input = []
        node_z_path_input = []
        current_length_input = 0

    if node_path_length > 2 and current_length_input > sampling_length_input:
        if node_pid in swcindex_nodeid_map_input.keys():
            current_path_input.append(node_pid)
        edge_path_list_input.append(current_path_input)
        for remove_i in range(0, node_path_length, 1):
            cur_node_id = current_path_input[remove_i]
            if remove_i == 0:
                if debug_downsample_swc:
                    print('if condition remove_i', remove_i)
                start_node_id = cur_node_id
                end_node_id = sampled_tree_map_input[start_node_id]
            elif remove_i == node_path_length - 1:
                if debug_downsample_swc:
                    print('elif condition remove_i', remove_i)
                    print('end_node_id', end_node_id)
                if end_node_id != -1:
                    end_node_id_parent = sampled_tree_map_input[cur_node_id]
                    sampled_tree_map_input.pop(cur_node_id)
                    sampled_tree_map_input[end_node_id] = end_node_id_parent
            else:
                if debug_downsample_swc:
                    print('else condition remove_i', remove_i)
                if cur_node_id in sampled_tree_map_input.keys():
                    sampled_tree_map_input.pop(cur_node_id)
                    if debug_downsample_swc:
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
        return get_sampled_edge_path(sampling_length_input=sampling_length_input,
                                     swc_array_input=swc_array_input,
                                     swcindex_nodeid_map_input=swcindex_nodeid_map_input,
                                     child_node_input=node_pid,
                                     current_path_input=current_path_input,
                                     node_x_path_input=node_x_path_input,
                                     node_y_path_input=node_y_path_input,
                                     node_z_path_input=node_z_path_input,
                                     current_length_input=current_length_input,
                                     edge_path_list_input=edge_path_list_input,
                                     tree_map_copy_input=tree_map_copy_input,
                                     sampled_tree_map_input=sampled_tree_map_input)
    else:
        return edge_path_list_input


def get_sampled_edge_path_v2(sampling_length_input,
                             swc_array_input,
                             swcindex_nodeid_map_input,
                             child_node_input,
                             current_path_input,
                             node_x_path_input,
                             node_y_path_input,
                             node_z_path_input,
                             current_length_input,
                             edge_path_list_input,
                             tree_map_copy_input,
                             sampled_tree_map_input):
    if child_node_input in tree_map_copy_input.keys():
        swc_array_row_number = swcindex_nodeid_map_input[child_node_input]
    else:
        if len(current_path_input) > 0:
            edge_path_list_input.append(current_path_input)
        current_path_input = []
        node_x_path_input = []
        node_y_path_input = []
        node_z_path_input = []
        current_length_input = 0
        child_node_input = list(tree_map_copy_input.keys())[0]
        if len(tree_map_copy_input) > 1:
            return get_sampled_edge_path_v2(sampling_length_input=sampling_length_input,
                                            swc_array_input=swc_array_input,
                                            swcindex_nodeid_map_input=swcindex_nodeid_map_input,
                                            child_node_input=child_node_input,
                                            current_path_input=current_path_input,
                                            node_x_path_input=node_x_path_input,
                                            node_y_path_input=node_y_path_input,
                                            node_z_path_input=node_z_path_input,
                                            current_length_input=current_length_input,
                                            edge_path_list_input=edge_path_list_input,
                                            tree_map_copy_input=tree_map_copy_input,
                                            sampled_tree_map_input=sampled_tree_map_input)
        else:
            return edge_path_list_input
        # swc_array_row_number = swcindex_nodeid_map_input[child_node_input]
    node_id = int(swc_array_input[swc_array_row_number, 0])
    node_x = swc_array_input[swc_array_row_number, 2]
    node_y = swc_array_input[swc_array_row_number, 3]
    node_z = swc_array_input[swc_array_row_number, 4]
    node_pid = int(swc_array_input[swc_array_row_number, 6])

    current_path_input.append(node_id)
    node_x_path_input.append(node_x)
    node_y_path_input.append(node_y)
    node_z_path_input.append(node_z)
    current_length = 0
    node_path_length = len(current_path_input)

    if len(current_path_input) > 1.5:
        for i in range(1, len(current_path_input)):
            current_length = (node_x_path_input[i] - node_x_path_input[i - 1]) * (
                    node_x_path_input[i] - node_x_path_input[i - 1]) + current_length
            current_length = (node_y_path_input[i] - node_y_path_input[i - 1]) * (
                    node_y_path_input[i] - node_y_path_input[i - 1]) + current_length
            current_length = (node_z_path_input[i] - node_z_path_input[i - 1]) * (
                    node_z_path_input[i] - node_z_path_input[i - 1]) + current_length
    current_length_input = current_length_input + math.sqrt(current_length)
    if current_length_input > 200:
        current_path_input = []
        node_x_path_input = []
        node_y_path_input = []
        node_z_path_input = []
        current_length_input = 0

    if node_path_length > 2 and current_length_input > sampling_length_input:
        if node_pid in swcindex_nodeid_map_input.keys():
            current_path_input.append(node_pid)
        edge_path_list_input.append(current_path_input)
        for remove_i in range(0, node_path_length, 1):
            cur_node_id = current_path_input[remove_i]
            if remove_i == 0:
                start_node_id = cur_node_id
                end_node_id = sampled_tree_map_input[start_node_id]
            elif remove_i == node_path_length - 1:
                if end_node_id != -1:
                    end_node_id_parent = sampled_tree_map_input[cur_node_id]
                    sampled_tree_map_input.pop(cur_node_id)
                    sampled_tree_map_input[end_node_id] = end_node_id_parent
            else:
                if cur_node_id in sampled_tree_map_input.keys():
                    sampled_tree_map_input.pop(cur_node_id)

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
        return get_sampled_edge_path_v2(sampling_length_input=sampling_length_input,
                                        swc_array_input=swc_array_input,
                                        swcindex_nodeid_map_input=swcindex_nodeid_map_input,
                                        child_node_input=node_pid,
                                        current_path_input=current_path_input,
                                        node_x_path_input=node_x_path_input,
                                        node_y_path_input=node_y_path_input,
                                        node_z_path_input=node_z_path_input,
                                        current_length_input=current_length_input,
                                        edge_path_list_input=edge_path_list_input,
                                        tree_map_copy_input=tree_map_copy_input,
                                        sampled_tree_map_input=sampled_tree_map_input)
    else:
        return edge_path_list_input


def get_newnode_path_from_newnode_id(newnode_id_input, newnode_paths_input):
    newnode_path_f = newnode_paths_input[newnode_id_input]
    return newnode_path_f


def edge_exist_newnode_path_pair(newnode_id1_input, newnode_id2_input, newnode_paths_input):
    newnode_path1_f = get_newnode_path_from_newnode_id(newnode_id1_input, newnode_paths_input)
    newnode_path2_f = get_newnode_path_from_newnode_id(newnode_id2_input, newnode_paths_input)
    if newnode_path1_f[-1] == newnode_path2_f[0]:
        return True
    elif newnode_path1_f[0] == newnode_path2_f[-1]:
        return True
    elif newnode_path1_f[0] == newnode_path2_f[0]:
        return True
    elif newnode_path1_f[-1] == newnode_path2_f[-1]:
        return True
    else:
        return False


def get_edge_index_from_newnodes(newnode_paths_input):
    total_node_number_f = len(newnode_paths_input)
    edge_counter = 0
    for node_i in range(0, total_node_number_f-1, 1):
        for node_j in range(node_i+1, total_node_number_f, 1):
            edge_exist = edge_exist_newnode_path_pair(newnode_id1_input=node_i,
                                                      newnode_id2_input=node_j,
                                                      newnode_paths_input=newnode_paths_input)
            if edge_exist:
                edge_counter = edge_counter + 1
                if edge_counter == 1:
                    edge_index_f = np.asarray([node_i, node_j])
                elif edge_counter > 1:
                    edge_index_f = np.vstack((edge_index_f, np.asarray([node_i, node_j])))
    return edge_index_f


def swc_to_branch_id(swc_array_input):
    parent_ids, parent_id_times = np.unique(swc_array_input[:, 6],
                                            return_counts=True)
    parent_id_branch_index = np.argwhere(parent_id_times > 1)
    parent_id_branch = parent_ids[parent_id_branch_index]
    parent_id_branch = parent_id_branch.astype('int')
    parent_id_branch_times = parent_id_times[parent_id_branch_index]
    return parent_id_branch, parent_id_branch_times


def show_swc_vis(tree_dict_input, title_input):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    for cur_node, cur_node_pid in tree_dict_input['tree_map'].items():
        if cur_node_pid in tree_dict_input['swcindex_nodeid_map'].keys():
            node_x_start, node_y_start, node_z_start = node_id_to_location(
                swc_array_input=tree_dict_input['swc_array'],
                node_id_input=cur_node,
                swcindex_nodeid_map_input=tree_dict_input[
                    'swcindex_nodeid_map'])
            node_x_end, node_y_end, node_z_end = node_id_to_location(swc_array_input=tree_dict_input['swc_array'],
                                                                     node_id_input=cur_node_pid,
                                                                     swcindex_nodeid_map_input=tree_dict_input[
                                                                         'swcindex_nodeid_map'])
            # print('node_x, node_y, node_z', node_x_end, node_y_end, node_z_end)
            ax.plot3D([node_x_start, node_x_end],
                      [node_y_start, node_y_end],
                      [node_z_start, node_z_end],
                      linewidth=1.4,
                      color=[0, 1, 0, 1])
    plt.title(title_input)


def show_edge_path_list(output_edge_path_list_input, tree_dict_input, title_input):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    for cur_node_path_i, cur_node_path in enumerate(output_edge_path_list_input):
        node_x_list = []
        node_y_list = []
        node_z_list = []
        for node_i, cur_node in enumerate(cur_node_path):
            node_x, node_y, node_z = node_id_to_location(swc_array_input=tree_dict_input['swc_array'],
                                                         node_id_input=cur_node,
                                                         swcindex_nodeid_map_input=tree_dict_input[
                                                             'swcindex_nodeid_map'])
            node_x_list.append(node_x)
            node_y_list.append(node_y)
            node_z_list.append(node_z)
        ax.plot3D(node_x_list,
                  node_y_list,
                  node_z_list,
                  color=[random.random(), random.random(), random.random(), 1])
    plt.title(title_input)


def show_edge_path_list_with_prob(output_edge_path_list_input, tree_dict_input, node_labels_input, title_input):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    for cur_node_path_i, cur_node_path in enumerate(output_edge_path_list_input):
        node_x_list = []
        node_y_list = []
        node_z_list = []
        node_prob = node_labels_input[cur_node_path_i]
        for node_i, cur_node in enumerate(cur_node_path):
            node_x, node_y, node_z = node_id_to_location(swc_array_input=tree_dict_input['swc_array'],
                                                         node_id_input=cur_node,
                                                         swcindex_nodeid_map_input=tree_dict_input[
                                                             'swcindex_nodeid_map'])
            node_x_list.append(node_x)
            node_y_list.append(node_y)
            node_z_list.append(node_z)
        ax.plot3D(node_x_list,
                  node_y_list,
                  node_z_list,
                  linewidth=1/(node_prob+0.1),
                  color=[random.random(), random.random(), random.random(), 1])
    plt.title(title_input)