from config.Config_testing import Config
from package.rivuletpy.rivuletpy.utils.io import loadswc
from package.rivuletpy.rivuletpy.utils.graph_helper import create_tree_map
from package.rivuletpy.rivuletpy.utils.graph_helper import swc_to_total_node_num
from package.rivuletpy.rivuletpy.utils.graph_helper import create_swcindex_nodeid_map
from package.rivuletpy.rivuletpy.utils.graph_helper2 import node_id_to_location
from package.rivuletpy.rivuletpy.utils.graph_helper4 import get_subs_from_test_list
from package.rivuletpy.rivuletpy.utils.graph_helper4 import get_heatmap_from_casename
from package.rivuletpy.rivuletpy.utils.graph_helper4 import get_gtswc_from_sub
from package.rivuletpy.rivuletpy.utils.graph_helper4 import get_newnodes_from_swc
from scipy.spatial.distance import cdist
import numpy as np
# from package.rivuletpy.rivuletpy.utils.graph_helper4 import newnode_all_help


def im_bound_check(im_sz_input, pt_input):
    if pt_input[0] < 0 or pt_input[1] < 0 or pt_input[2] < 0:
        bound_flag = False
    elif pt_input[0] > im_sz_input[0] or pt_input[1] > im_sz_input[1] or pt_input[2] > im_sz_input[2]:
        bound_flag = False
    else:
        bound_flag = True
    return bound_flag


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


def get_gt_infor(gt_swc):
    gt_total_node_number = swc_to_total_node_num(swc_array_input=gt_swc)
    gt_tree_map = create_tree_map(swc_array_input=gt_swc, total_node_number_input=gt_total_node_number)
    gt_swcindex_nodeid_map = create_swcindex_nodeid_map(swc_array_input=gt_swc)
    gt_newnodes = get_newnodes_from_swc(swc_array_input=gt_swc)
    print('---Ground Truth New Nodes Done---')
    for cur_gt_newnode in gt_newnodes:
        node_x1, node_y1, node_z1 = node_id_to_location(swc_array_input=gt_swc,
                                                        node_id_input=cur_gt_newnode[6],
                                                        swcindex_nodeid_map_input=gt_swcindex_nodeid_map)
        node_x2, node_y2, node_z2 = node_id_to_location(swc_array_input=gt_swc,
                                                        node_id_input=cur_gt_newnode[7],
                                                        swcindex_nodeid_map_input=gt_swcindex_nodeid_map)
        if cur_gt_newnode[6] not in gt_tree_map.keys() or cur_gt_newnode[7] not in gt_tree_map.keys():
            print('cur_gt_newnode out of tree_map', cur_gt_newnode)
    print('---Ground Truth New Nodes Locations Done---')
    print(node_x1, node_y1, node_z1)
    print(node_x2, node_y2, node_z2)
    return None


# thre = 0.1
# config = Config()
# subs = get_subs_from_test_list(config_input=config)
# sub = subs[0]
# heatmap, heatmap_bi = get_heatmap_from_casename(sub_input=sub, config_input=config)
# print('---Heatmap Done---')




