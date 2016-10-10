import numpy as np
from scipy.spatial.distance import cdist

def precision_recall(swc1, swc2, dist1=4, dist2=2):
    '''
    Calculate the precision, recall and F1 score between swc1 and swc2 (ground truth)
    It generates a new swc file with node types indicating the agreement between two input swc files
    In the output swc file: node type - 1. the node is in both swc1 agree with swc2
                                                        - 2. the node is in swc1, not in swc2 (over-traced)
                                                        - 3. the node is in swc2, not in swc1 (under-traced)
    target: The swc from the tracing method
    gt: The swc from the ground truth
    dist1: The distance to consider for precision
    dist2: The distance to consider for recall
    '''

    TPCOLOUR, FPCOLOUR, FNCOLOUR  = 3, 2, 180 # COLOUR is the SWC node type defined for visualising in V3D

    d = cdist(swc1[:, 2:5], swc2[:, 2:5])
    mindist1 = d.min(axis=1)
    tp = (mindist1 < dist1).sum()
    fp = swc1.shape[0] - tp

    mindist2 = d.min(axis=0)
    fn = (mindist2 > dist2).sum()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)

    # Make the swc for visual comparison
    swc1[mindist1 <= dist1, 1] = TPCOLOUR
    swc1[mindist1 > dist1, 1] = FPCOLOUR
    swc2_fn = swc2[mindist2 > dist2, :]
    swc2_fn[:, 0] = swc2_fn[:, 0] + 100000
    swc2_fn[:, -1] = swc2_fn[:, -1] + 100000
    swc2_fn[:, 1] = FNCOLOUR
    swc_compare = np.vstack((swc1, swc2_fn))
    swc_compare[:, -2]  = 1

    return (precision, recall, f1), swc_compare


def gaussian_distance(swc1, swc2, sigma=2.):
    '''
    The geometric metrics of NetMets. The gaussian distances between the closest neighbours
    returns : (M1, M2) where 
    D. Mayerich, C. Bjornsson, J. Taylor, and B. Roysam, 
    “NetMets: software for quantifying and visualizing errors in biological network segmentation.,” 
    BMC Bioinformatics, vol. 13 Suppl 8, no. Suppl 8, p. S7, 2012.
    '''

    d = cdist(swc1[:, 2:5], swc2[:, 2:5]) # Pairwise distances between 2 swc files
    mindist1 = d.min(axis=1)
    M1 = 1 - np.exp(mindist1 ** 2  / (2 * sigma ** 2))
    mindist2 = d.min(axis=0)
    M2 = 1 - np.exp(mindist2 ** 2  / (2 * sigma ** 2))
    return M1, M2


