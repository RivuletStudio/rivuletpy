import numpy as np
import math
from random import random, randrange
from collections import Counter
from scipy.spatial.distance import cdist
from .utils.io import saveswc

class SWC(object):
    def __init__(self, soma=None):
        self._data = np.zeros((1,8)) 
        if soma:
            self._data[0, :] = np.asarray([0, 1, soma.centroid[0], soma.centroid[1], soma.centroid[2], soma.radius, -1, 1])

    def add(self, swc_nodes):
        np.vstack((self._data, swc_nodes))

    def add_branch(self, branch, pidx=None, random_color=True):
        '''
        Add a branch to swc.
        Note: This swc is special with N X 8 shape. The 8-th column is the online confidence
        '''
        if random_color:
            rand_node_type = randrange(256)

        new_branch = np.zeros((len(branch.pts), 8))
        id_start = 1 if self._data.shape[0] == 1 else self._data[:, 0].max() + 1

        for i in range(len(branch.pts)):
            p, r, c = branch.pts[i], branch.radius[i], branch.conf[i]
            id = id_start + i
            # 3 for basal dendrite; 4 for apical dendrite;
            # However now we cannot differentiate them automatically
            nodetype = 3

            if i == len(branch.pts) - 1:  # The end of this branch
                pid = self._data[pidx, 0] if pidx is not None else -2
                if pid is not -2 and pid != 0 and self._data.shape[0] != 1:
                    # Its connected node is fork point
                    self._data[self._data[:, 0] == pid, 1] = 5  
            else:
                pid = id_start + i + 1
                if i == 0:
                    nodetype = 6  # Endpoint

            assert(pid != id)
            new_branch[i] = np.asarray([
                id, rand_node_type
                if random_color else nodetype, p[0], p[1], p[2], r, pid, c])

        # Check if any tail should be connected to its tail
        tail = new_branch[0]
        matched, minidx = self.match(tail[2:5], tail[5])
        if matched and self._data[minidx, 6] is -2:
            self._data[minidx, 6] = tail[0]

        self._data = np.vstack((self._data, new_branch))

    def _prune_leaves(self):
        # Find all the leaves
        childctr = Counter(self._data[:, 6])
        leafidlist = [id for id in self._data[:, 0]
                      if id not in self._data[:, 6]]
        id2dump = []
        rmean = self._data[:,5].mean() # Mean radius

        for leafid in leafidlist:  # Iterate each leaf node
            nodeid = leafid
            branch = []
            while True:  # Get the leaf branch out
                node = self._data[self._data[:, 0] == nodeid, :].flatten()
                if node.size == 0:
                    break
                branch.append(node)
                parentid = node[6]
                if childctr[parentid] is not 1:
                    break  # merged / unconnected
                nodeid = parentid

            # Get the length of the leaf
            leaflen = sum([
                np.linalg.norm(branch[i][2:5] - branch[i - 1][2:5])
                for i in range(1, len(branch))
            ])

            # Prune if the leave is too short or
            # the confidence of the leave branch is too low
            if leaflen <= 4 * rmean:
                id2dump.extend([node[0] for node in branch])

        # Only keep the swc nodes not in the dump id list
        cutted = []
        for nodeidx in range(self._data.shape[0]):
            if self._data[nodeidx, 0] not in id2dump:
                cutted.append(self._data[nodeidx, :])

        cutted = np.squeeze(np.dstack(cutted)).T
        self._data = cutted

    def _prune_unreached(self):
        '''
        Only keep the largest connected component
        '''
        swcdict = {}
        for n in self._data:  # Hash all the swc nodes
            swcdict[n[0]] = Node(n[0])

        # Try to join all the unconnected branches at first
        for i, n in enumerate(self._data):
            if n[6] not in swcdict:
                # Try to match it
                matched, midx = self.match(n[2:5], n[5])
                if matched:
                    self._data[i, 6] = self._data[midx, 0]

        # Add mutual links for all nodes
        for n in self._data:
            id = n[0]
            pid = n[6]
            if pid >= 0:
                swcdict[id].add_link(swcdict[pid])

        groups = connected_components(set(swcdict.values()))
        lenlist = [len(g) for g in groups]
        maxidx = lenlist.index(max(lenlist))
        set2keep = groups[maxidx]
        id2keep = [n.id for n in set2keep]
        self._data = self._data[np.in1d(self._data[:, 0], np.asarray(id2keep)), :]

    def prune(self):
        self._prune_unreached()
        self._prune_leaves()

    def reset(self, crop_region, zoom_factor):
        '''
        Pad and rescale swc back to the original space
        '''

        tswc = self._data.copy()
        if zoom_factor != 1.:  # Pad the swc back to original space
            tswc[:, 2:5] *= 1. / zoom_factor

        # Pad the swc back
        tswc[:, 2] += crop_region[0, 0]
        tswc[:, 3] += crop_region[1, 0]
        tswc[:, 4] += crop_region[2, 0]
        self._data = tswc

    def get_id(self, idx):
        return self._data[idx, 0]


    def match(self, pos, radius):
        '''
        Find the closest ground truth node 
        '''

        nodes = self._data[:, 2:5]
        distlist = np.squeeze(cdist(pos.reshape(1, 3), nodes))
        if distlist.size == 0:
            return False, -2
        minidx = distlist.argmin()
        minnode = self._data[minidx, 2:5]

        # See if either of them can cover each other with a ball of their own radius
        mindist = np.linalg.norm(pos - minnode)
        return radius > mindist or self._data[minidx, 5] > mindist, minidx

    def size(self):
        return self._data.shape[0]

    def save(self, fname):
        saveswc(fname, self._data)

    def view(self):
        from rivuletpy.utils.rendering3 import Viewer3, Line3

        # Compute the center of mass
        center = self._data[:,2:5].mean(axis=0)
        translated = self._data[:,2:5] - np.tile(center, (self._data.shape[0], 1))

        # Init viewer
        viewer = Viewer3(800,800,800)
        viewer.set_bounds(self._data[:, 2].min(), self._data[:, 2].max(),
                          self._data[:, 3].min(), self._data[:, 3].max(),
                          self._data[:, 4].min(), self._data[:, 4].max())
        lid = self._data[:,0]

        line_color = [random(), random(), random()]
        for i in range(self._data.shape[0]):
            # Change color if its a bifurcation 
            if (self._data[i, 0] == self._data[:, -1]).sum() > 1:
                line_color = [random(), random(), random()]

            # Draw a line between this node and its parent
            if i < self._data.shape[0] - 1 and self._data[i, 0] == self._data[i+1, -1]:
                l = Line3(translated[i, :], translated[i+1, :])
                l.set_color(*line_color)
                viewer.add_geom(l)
            else:
                pid = self._data[i, -1]
                pidx = np.argwhere(pid == lid).flatten()
                if len(pidx) == 1:
                    l = Line3(translated[i, :], translated[pidx, :].flatten())
                    l.set_color(*line_color)
                    viewer.add_geom(l)

        while(True):
            try:
                viewer.render(return_rgb_array=False)
            except KeyboardInterrupt:
                break


def get_subtree_nodeids(swc, node):
    subtreeids = np.array([])

    # Find children
    chidx = np.argwhere(node[0] == swc[:, 6])

    # Recursion stops when there this node is a
    # leaf with no children, return itself
    if chidx.size == 0:
        return node[0]
    else:
        # Get the node ids of each children
        for c in chidx:
            subids = get_subtree_nodeids(swc, swc[c, :].squeeze())
            subtreeids = np.hstack((subtreeids, subids, node[0]))

    return subtreeids




class Node(object):
    def __init__(self, id):
        self.__id = id
        self.__links = set()

    @property
    def id(self):
        return self.__id

    @property
    def links(self):
        return set(self.__links)

    def add_link(self, other):
        self.__links.add(other)
        other.__links.add(self)


def connected_components(nodes):
    '''
    The function to look for connected components.
    Reference: https://breakingcode.wordpress.com/2013/04/08/finding-connected-components-in-a-graph/
    '''

    # List of connected components found. The order is random.
    result = []

    # Make a copy of the set, so we can modify it.
    nodes = set(nodes)

    # Iterate while we still have nodes to process.
    while nodes:

        # Get a random node and remove it from the global set.
        n = nodes.pop()

        # This set will contain the next group of nodes
        # connected to each other.
        group = {n}

        # Build a queue with this node in it.
        queue = [n]

        # Iterate the queue.
        # When it's empty, we finished visiting a group of connected nodes.
        while queue:

            # Consume the next item from the queue.
            n = queue.pop(0)

            # Fetch the neighbors.
            neighbors = n.links

            # Remove the neighbors we already visited.
            neighbors.difference_update(group)

            # Remove the remaining nodes from the global set.
            nodes.difference_update(neighbors)

            # Add them to the group of connected nodes.
            group.update(neighbors)

            # Add them to the queue, so we visit them in the next iterations.
            queue.extend(neighbors)

        # Add the group to the list of groups.
        result.append(group)

    # Return the list of groups.
    return result
