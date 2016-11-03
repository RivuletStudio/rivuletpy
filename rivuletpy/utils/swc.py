import numpy as np
import math
from scipy.spatial.distance import cdist


def reset_swc(swc, crop_region, zoom_factor):
    '''
    Pad and rescale swc back to the original space
    '''

    tswc = swc.copy()
    if zoom_factor != 1.:  # Pad the swc back to original space
        tswc[:, 2:5] *= 1. / zoom_factor

    # Pad the swc back
    tswc[:, 2] += crop_region[0, 0]
    tswc[:, 3] += crop_region[1, 0]
    tswc[:, 4] += crop_region[2, 0]
    return tswc


def swcradius(bimg, swc):
    for i in range(swc.shape[0]):
        swc[i, 5] = getradius(bimg, swc[i, 2], swc[i, 3], swc[i, 4])
    return swc


def getradius(bimg, x, y, z):
    r = 0
    x = math.floor(x)
    y = math.floor(y)
    z = math.floor(z)

    while True:
        r += 1
        try:
            if bimg[max(x - r, 0):min(x + r + 1, bimg.shape[0]), max(y - r, 0):
                    min(y + r + 1, bimg.shape[1]), max(z - r, 0):min(
                        z + r + 1, bimg.shape[2])].sum() / (2 * r + 1)**3 < .6:
                break
        except IndexError:
            break

    return r


def flipswc(swc, axis=1):
    '''
    Flip the swc according to one axis.
    Needed when the image was read in a wrong order
    '''
    p = swc[:, 2 + axis]
    swc[:, 2 + axis] = np.abs(p - p.max())
    return swc


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


def cleanswc(swc, radius=True):
    '''
    Only keep the largest connected component
    '''
    swcdict = {}
    for n in swc:  # Hash all the swc nodes
        swcdict[n[0]] = Node(n[0])

    # Try to join all the unconnected branches at first
    for i, n in enumerate(swc):
        if n[6] not in swcdict:
            # Try to match it
            matched, midx = match_r1(swc, n[2:5], n[5], 1.5)
            if matched:
                swc[i, 6] = swc[midx, 0]

    # Add mutual links for all nodes
    for n in swc:
        id = n[0]
        pid = n[6]
        if pid >= 0:
            swcdict[id].add_link(swcdict[pid])

    groups = connected_components(set(swcdict.values()))
    lenlist = [len(g) for g in groups]
    maxidx = lenlist.index(max(lenlist))
    set2keep = groups[maxidx]
    id2keep = [n.id for n in set2keep]
    swc = swc[np.in1d(swc[:, 0], np.asarray(id2keep)), :]
    if not radius:
        swc[:, 5] = 1

    return swc


def match_r1(swc, pos, radius, wiring):
    '''
    The node match used by Rivulet1 which uses a wiring threshold
    Deprecated in the standard Rivulet pipeline 
    Used only for experiments
    '''

    # Find the closest ground truth node 
    nodes = swc[:, 2:5]
    distlist = np.squeeze(cdist(pos.reshape(1, 3), nodes))
    if distlist.size == 0:
        return False, -2
    minidx = distlist.argmin()
    minnode = swc[minidx, 2:5]

    # See if either of them can cover each other with a ball of their own radius
    mindist = np.linalg.norm(pos - minnode)
    return radius > wiring * mindist or swc[minidx,
                                            5] * wiring > mindist, minidx


def match(swc, pos, radius):
    '''
    Find the closest ground truth node 
    '''

    nodes = swc[:, 2:5]
    distlist = np.squeeze(cdist(pos.reshape(1, 3), nodes))
    if distlist.size == 0:
        return False, -2
    minidx = distlist.argmin()
    minnode = swc[minidx, 2:5]

    # See if either of them can cover each other with a ball of their own radius
    mindist = np.linalg.norm(pos - minnode)
    return radius > mindist or swc[minidx, 5] > mindist, minidx