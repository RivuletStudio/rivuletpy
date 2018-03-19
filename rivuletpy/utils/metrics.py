from collections import deque
import numpy as np
from scipy.spatial.distance import cdist

def precision_recall(swc1, swc2, dist1=4, dist2=4):
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

    # Compute the SD, SSD, SSD% defined in Peng.et.al 2010
    SD = (np.mean(mindist1) + np.mean(mindist2)) / 2
    far1, far2 = mindist1[mindist1 > dist1], mindist2[mindist2 > dist2]   
    SSD = (np.mean(far1) + np.mean(far2)) / 2
    pSSD = (len(far1) / len(mindist1) + len(far2) / len(mindist2)) / 2

    return (precision, recall, f1), (SD, SSD, pSSD), swc_compare


def upsample_swc(swc):

    tswc = swc.copy()

    id_idx = {}
    # Build a nodeid->idx hash table
    for nodeidx in range(tswc.shape[0]):
        id_idx[tswc[nodeidx, 0]] = nodeidx

    newid = tswc[:,0].max() + 1
    newnodes = []
    for nodeidx in range(tswc.shape[0]):
        pid = tswc[nodeidx, -1] # parent id

        if pid not in id_idx:
            # raise Exception('Parent with id %d not found' % pid)
            continue

        nodepos = tswc[nodeidx, 2:5]
        parentpos = tswc[id_idx[pid], 2:5]

        if np.linalg.norm(nodepos - parentpos) > 1.: # Add a node in the middle if too far
            mid_pos = nodepos + 0.5 * (parentpos - nodepos)
            newnodes.append( np.asarray([newid, 2, mid_pos[0], mid_pos[1], mid_pos[2], 1, pid]) )
            newid += 1
            tswc[nodeidx, -1] = newid

    # Stack the new nodes to the end of the swc file
    newnodes = np.vstack(newnodes)
    tswc = np.vstack((tswc, newnodes))
    return tswc


def gaussian_distance(swc1, swc2, sigma=2.):
    '''
    The geometric metrics of NetMets. The gaussian distances between the closest neighbours
    returns : (M1, M2) where M1 is the gaussian distances from the nodes in swc1 to their closest neighbour in swc2;
    vise versa for M2

    D. Mayerich, C. Bjornsson, J. Taylor, and B. Roysam, 
    “NetMets: software for quantifying and visualizing errors in biological network segmentation.,” 
    BMC Bioinformatics, vol. 13 Suppl 8, no. Suppl 8, p. S7, 2012.
    '''
    swc1 = upsample_swc(swc1)
    swc2 = upsample_swc(swc2)

    d = cdist(swc1[:, 2:5], swc2[:, 2:5]) # Pairwise distances between 2 swc files
    mindist1 = d.min(axis=1)
    M1 = 1 - np.exp(mindist1 ** 2  / (2 * sigma ** 2))
    mindist2 = d.min(axis=0)
    M2 = 1 - np.exp(mindist2 ** 2  / (2 * sigma ** 2))
    return M1, M2


def connectivity_distance(swc1, swc2, sigma=2., ignore_leaf=True):
    '''
    The connectivity metrics of NetMets. 
    Returns (midx1, midx2): the indices of nodes in each swc that have connection errors

    D. Mayerich, C. Bjornsson, J. Taylor, and B. Roysam, 
    “NetMets: software for quantifying and visualizing errors in biological network segmentation.,” 
    BMC Bioinformatics, vol. 13 Suppl 8, no. Suppl 8, p. S7, 2012.
    '''

    # graph Initialisation
    d = cdist(swc1[:, 2:5], swc2[:, 2:5]) # Pairwise distances between 2 swc files
    mindist1, mindist2 = d.min(axis=1), d.min(axis=0)
    minidx1, minidx2 = d.argmin(axis=1), d.argmin(axis=0)

    # Colour nodes - matched nodes have the same colour
    cnodes1, cnodes2 = {}, {}# Coloured Nodes <id, colour>
    for i in range(swc1.shape[0]):
        if mindist1[i] < sigma: 
            cnodes1[swc1[i, 0]] = i
            cnodes2[swc2[minidx1[i], 0]] = i

    # Build Initial graphs, Edge: <id_i, id_j>: 1
    g1 = build_graph_from_swc(swc1) 
    g2 = build_graph_from_swc(swc2)

    # BFS to build the core graph for both swc, returns the remaining edges not used to build the core graph
    dg1 = build_core_graph(g1, cnodes1) 
    dg2 = build_core_graph(g2, cnodes2) 

    # Find the diff edges with coloured nodes involved 
    mid1 = set() 
    for id in dg1:
        for nid in g1[id]: 
            if nid in cnodes1: mid1.add(nid)

    mid2 = set() 
    for id in dg2:
        for nid in g2[id]: 
            if nid in cnodes2: mid2.add(nid)

    id_idx_hash1 = {}
    for i in range(swc1.shape[0]): id_idx_hash1[swc1[i, 0]] = i

    id_idx_hash2 = {}
    for i in range(swc2.shape[0]): id_idx_hash2[swc2[i, 0]] = i

    midx1 = [ int(id_idx_hash1[id]) for id in mid1 ] # Mistake coloured nodes in edges of dg1
    midx2 = [ int(id_idx_hash2[id]) for id in mid2 ] # Mistake coloured nodes in edges of dg2

    # Filter out the midx of nodes on leaf segments
    if ignore_leaf:
        leafidx1 = find_leaf_idx(swc1)
        midx1 = set(midx1) - set(leafidx1)
        leafidx2 = find_leaf_idx(swc2)
        midx2 = set(midx2) - set(leafidx2)

    return len(midx1) / len(mid1), len(midx2) / len(mid2)


def find_leaf_idx(swc):
    # The degree of a node is the number of children + 1 except the root
    degree  = np.zeros(swc.shape[0])
    for i  in range(swc.shape[0]):
        degree[i] = np.count_nonzero(swc[:, -1] == swc[i, 0]) + 1

    # A node is a leaf node if it is parent to no other node
    leaf_segment_idx = []
    leaf_node_idx = np.where(degree == 1)[0]
    for idx in leaf_node_idx:
        # Add its parent to the leaf segment idx list if its parent degree < 3
        nodeidx = idx
        while degree[nodeidx] < 3:
            leaf_segment_idx.append(int(nodeidx))
            if swc[nodeidx, -1] < 0:
                break
            nodeidx = np.where(swc[:, 0] == swc[nodeidx, -1])[0]

    return leaf_segment_idx


def build_graph_from_swc(swc):
    g = {}
    for i in range(swc.shape[0]):
        id, pid = swc[i, 0], swc[i, -1]

        if id in g:
            g[id].append(pid)
        else:
            g[id] = [pid]

        if pid in g:
            g[pid].append(id)
        else:
            g[pid] = [id]

    for key, value in g.items():
        g[key] = set(value)

    return g


def build_core_graph(g, cnodes):
    '''
    Returns the edges not used in building the core graph (topologically matched between two graphs)
    '''

    cnodes = cnodes.copy() # Coloured node list to mark which have not been discovered 
    dg = g.copy()

    while cnodes:
        root = next(iter(cnodes))
        core_neighbours = find_core_neighbours_bfs(dg, root, cnodes)  # BFS to discover the neighbour

        nodes_on_path = set()
        if  core_neighbours:
            for id in core_neighbours:
                nodes_on_path = nodes_on_path.union(track_path_nodes_dijstra(dg, id, root))
        else:
            nodes_on_path.add(root)

        cnodes.pop(root) # Remove the discovered coloured nodes
        for n in nodes_on_path:
            dg.pop(n, None) 

        for n in dg:
            dg[n] = dg[n].difference(nodes_on_path)

    return dg


def find_core_neighbours_bfs(g, root, cnodes):
    '''
    Find the coloured neighbours of root node with BFS search
    '''

    visited = {}
    node_queue = deque()
    visited[root] = True
    node_queue.append(root)
    core_neighbours = []

    while node_queue:
        r = node_queue.popleft()

        if r in cnodes and r != root: 
            core_neighbours.append(r) # If this node is coloured, bfs stops on it and add it to the core neighbours
        else:
            for n in g[r]: # visit all the neighbours of r
                if n not in visited:
                    visited[n] = True
                    node_queue.append(n)

    return core_neighbours


def track_path_nodes_dijstra(g, target, source):
    path = {}
    visited = {source: 0}
    nodes = g.copy()

    while nodes:
        min_node = None
        for node in nodes:
            if node in visited:
                if min_node is None:
                    min_node = node
                elif visited[node] < visited[min_node]:
                    min_node = node

        if min_node is None:
            break

        nodes.pop(min_node)
        tweight = visited[min_node]
        for n in g[min_node]:
            weight = tweight + 1
            if n not in visited or weight < visited[n]:
                visited[n] = weight
                path[n]  = min_node

        if min_node == target:
            break

    nodes_on_path, n = set(), target
    while n != source:
        n = path[n]
        nodes_on_path.add(n)

    return nodes_on_path
