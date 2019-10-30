from package.rivuletpy.rivuletpy.utils.io import loadswc
from package.rivuletpy.rivuletpy.utils.graph_helper import swc_to_total_node_num
from package.rivuletpy.rivuletpy.utils.graph_helper import create_swcindex_nodeid_map
from package.rivuletpy.rivuletpy.utils.graph_helper import create_tree_map
from package.rivuletpy.rivuletpy.utils.graph_helper import swc_to_branch_id
from package.rivuletpy.rivuletpy.utils.graph_helper2 import get_node_id_pid
from package.rivuletpy.rivuletpy.utils.graph_helper2 import edge_exist
from package.rivuletpy.rivuletpy.utils.graph_helper2 import edge_to_node
from package.rivuletpy.rivuletpy.utils.graph_helper3_v2 import reverse_tree_map
from package.rivuletpy.rivuletpy.utils.graph_helper3_v2 import find_node_pid
from package.rivuletpy.rivuletpy.utils.graph_helper3_v2 import create_all_newnode
from package.rivuletpy.rivuletpy.utils.graph_helper3_v2 import node_to_edge
from package.rivuletpy.rivuletpy.utils.graph_helper3_v2 import edge_vec_to_edge_nodes
from package.rivuletpy.rivuletpy.utils.metrics import precision_recall
from run_evaluating_tree import vtk2swc
from config.Config_testing import Config
from patch_extract import load_ct_im_dh
import numpy as np
import os


def create_edge_list(swc_array_input,
                     swcindex_nodeid_map_input,
                     tree_map_input,
                     reversed_tree_map_input,
                     newnode_all_input):
    cur_node_counter = 0
    for cur_node in tree_map_input.keys():
        edge_vec = node_to_edge(swc_array_input=swc_array_input,
                                node_id_input=cur_node,
                                swcindex_nodeid_map_input=swcindex_nodeid_map_input,
                                reversed_tree_map_input=reversed_tree_map_input,
                                newnode_all_input=newnode_all_input)
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


def newnode_all_help():
    print('newnode_id_input(0)',
          'new_node_structure(1)',
          'new_node_x(2)',
          'new_node_y(3)',
          'new_node_z(4)',
          'new_node_radius(5)',
          'node_id_input(6)',
          'node_pid_input(7)')


def get_heatmap_from_casename(sub_input):
    thre = 0.1
    liver_vessel_pred_file = os.path.join(config.test_data_path,
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


def get_subs_from_test_list(config_input):
    list_txt = config_input.test_list_txt
    if not os.path.isfile(list_txt):
        raise ValueError('the list file does not exist!')
    subs = [sub.rstrip('\n').rstrip('\r') for sub in open(list_txt, 'r').readlines()]
    return subs


def get_gtswc_from_sub(sub_input):
    save_tracing_path = './dh_test_bk_pts_online/tracing_eval'
    gt_vtk_tree_path = os.path.join(save_tracing_path,
                                    'pt_vessel_gt',
                                    sub_input)
    tree_names = os.listdir(gt_vtk_tree_path)
    tree_counter = 0
    for tree in (tree_names):
        if tree.endswith(".vtk"):
            if tree_counter == 0:
                tree1_path = gt_vtk_tree_path + '/' + tree
                print(tree1_path)
                gt_swc = vtk2swc(tree1_path)
            if tree_counter > 0:
                tree_path = gt_vtk_tree_path + '/' + tree
                gt2_swc = vtk2swc(tree_path)
                rand_i = a = np.random.randint(low=2,
                                               high=30)
                gt2_swc[:, 0] = gt2_swc[:, 0] + 80000 * rand_i
                gt2_swc[:, 6] = gt2_swc[:, 6] + 80000 * rand_i
                gt_swc = np.concatenate((gt_swc, gt2_swc), axis=0)
            tree_counter = tree_counter + 1
    return gt_swc


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


def unit_helper4():
    print('Unit Test begins')
    thre = 0.1
    config = Config()
    subs = get_subs_from_test_list(config_input=config)
    for sub in subs:
        print('the current case being processed is', sub)
    heatmap, heatmap_bi = get_heatmap_from_casename(sub_input=sub)
    print('the shape of heatmap', heatmap.shape)
    one_node_id = 6
    print('one_node_id',
          one_node_id)
#     one_node_feature = get_node_feature_from_heatmap(newnode_id_input=one_node_id,
#                                                      newnode_all_input=newnode_all,
#                                                      heatmap_input=heatmap)
#     print('one_node_feature.shape', one_node_feature.shape)
    return None


def unit_helper4_part2():
    # swcpath = '/home/donghao/Desktop/Liver_vessel/dh_test/runs/vessel_pred_0.1.swc'
    swcpath = './dh_test_bk_pts_online/tracing_eval/pt_vessel_pred/fe131b5c-d32a-4a58-9214-b327021f63a8/vessel_pred_0.1.swc'
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
    print('prediction newnode', newnode_all.shape)
    edge_list = create_edge_list(swc_array_input=swc_copy,
                                 swcindex_nodeid_map_input=swcindex_nodeid_map,
                                 reversed_tree_map_input=reversed_tree_map,
                                 newnode_all_input=newnode_all)
    edge_list = post_processing_edge_list(edge_list_input=edge_list)
    newnode_all_help()
    return newnode_all


def unit_helper4_part3():
    thre = 0.1
    config = Config()
    subs = get_subs_from_test_list(config_input=config)
    sub = subs[0]
    heatmap, heatmap_bi = get_heatmap_from_casename(sub_input=sub)
    gt_swc = get_gtswc_from_sub(sub_input=sub)
    gt_newnodes = get_newnodes_from_swc(swc_array_input=gt_swc)
#     (precision, recall, f1), (SD, SSD, pSSD), swc_compare = precision_recall(swc_copy,
#                                                                              gt_swc,
#                                                                              dist1=4,
#                                                                              dist2=4)
    return gt_newnodes