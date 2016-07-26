import numpy as np
import random
import math
from euclid import Point3
from scipy.spatial.distance import cdist

def gd(srcpt, ginterp, t, stepsize):
    gvec = np.asarray([g(srcpt)[0] for g in ginterp])
    if np.linalg.norm(gvec) <= 0: 
        return np.array([-1, -1, -1])
    gvec /= np.linalg.norm(gvec)
    srcpt -= stepsize * gvec
    return srcpt


def rk4(srcpt, ginterp, t, stepsize):
    # Compute K1
    k1 = np.asarray([g(srcpt)[0] for g in ginterp])

    if np.linalg.norm(k1) == 0:
        print('== gradient is zero at', srcpt) 
    k1 /= np.linalg.norm(k1)
    k1 *= stepsize
    tp = srcpt - 0.5 * k1 # Position of temporary point
    if not inbound(tp, t.shape):
        return srcpt

    # Compute K2
    k2 = np.asarray([g(tp)[0] for g in ginterp])
    k2 /= np.linalg.norm(k2)
    k2 *= stepsize
    tp = srcpt - 0.5 * k2 # Position of temporary point
    if not inbound(tp, t.shape):
        return srcpt

    # Compute K3
    k3 = np.asarray([g(tp)[0] for g in ginterp])
    k3 /= np.linalg.norm(k3)
    k3 *= stepsize
    tp = srcpt - k3 # Position of temporary point
    if not inbound(tp, t.shape):
        return srcpt

    # Compute K4
    k4 = np.asarray([g(tp)[0] for g in ginterp])
    k4 /= np.linalg.norm(k4)
    k4 *= stepsize

    # Compute final point
    endpt = srcpt - (k1 + k2*2 + k3*2 + k4)/6.0
    if not inbound(tp, t.shape):
        return endpt

    return endpt


def getradius(bimg, x, y, z):
    r = 0
    x = math.floor(x)   
    y = math.floor(y)   
    z = math.floor(z)   

    while True:
        r += 1
        try:
            if bimg[max(x-r, 0) : min(x+r+1, bimg.shape[0]),
                    max(y-r, 0) : min(y+r+1, bimg.shape[1]), 
                    max(z-r, 0) : min(z+r+1, bimg.shape[2])].sum() / (2*r + 1)**3 < 0.8:
                break
        except IndexError:
            break

    return r


def inbound(pt, shape):
    return all([True if 0 <= p <= s-1 else False for p,s in zip(pt, shape)])


def fibonacci_sphere(samples=1, randomize=True):
    rnd = 1.
    if randomize:
        rnd = random.random() * samples

    points = []
    offset = 2./samples
    increment = math.pi * (3. - math.sqrt(5.));

    for i in range(samples):
        y = ((i * offset) - 1) + (offset / 2);
        r = math.sqrt(1 - pow(y,2))

        phi = ((i + rnd) % samples) * increment

        x = math.cos(phi) * r
        z = math.sin(phi) * r

        points.append(np.array([x, y, z]))

    return points


def match(swc, pos, radius): 
    # Find the closest ground truth node 
    nodes = swc[:, 2:5]
    distlist = np.squeeze(cdist(pos.reshape(1,3), nodes))
    minidx = distlist.argmin()
    minnode = swc[minidx, 2:5]

    # See if either of them can cover each other with a ball of their own radius
    mindist = np.linalg.norm(pos - minnode)
    return radius > mindist or swc[minidx, 5] > mindist, minidx


def add2swc(swc, path, radius, connectid = None): 
    newbranch = np.zeros((len(path), 7))
    if swc is None: # It is the first branch to be added
        idstart = 1
    else:
        idstart = swc[:, 0].max() + 1

    for i, p in enumerate(path):
        id = idstart+i

        if i == len(path) - 1: # The end of this branch
            pid = -2 if connectid is None else connectid
        else:
            pid = idstart + i + 1

        newbranch[i] = np.asarray([id, 2, p[0], p[1], p[2], radius[i], pid])

    if swc is None:
        swc = newbranch
    else:
        # Check if any tail should be connected to its head
        head = newbranch[0]
        matched, minidx = match(swc, head[2:5], head[5])
        if matched and swc[minidx, -1] is -2: swc[minidx, -1] = head[0]
        swc = np.vstack((swc, newbranch))

    return swc


def constrain_range(min, max, minlimit, maxlimit):
    return list(range(min if min > minlimit else minlimit, max if max < maxlimit else maxlimit))


def get_subtree_nodeids(swc, node):
    subtreeids = np.array([])

    # Find children
    # print('-- Node here:', node)
    chidx = np.argwhere(node[0] == swc[:, 6])

    # Recursion stops when there this node is a leaf with no children, return itself 
    if chidx.size == 0:
        # print('== No Child, returning', node[0])
        return node[0]
    else:
        # print('== Got child')
        # Get the node ids of each children
        for c in chidx:
            subids = get_subtree_nodeids(swc, swc[c, :].squeeze())
            # print('==Trying to append', subtreeids, subids, node[0])
            subtreeids = np.hstack((subtreeids, subids, node[0]))

    # print('==Returning:', subtreeids)
    return subtreeids


def cleanswc(swc):
    unconnected_idx = np.argwhere(swc[:, -1] == -2)
    nodes2del = np.array([]) 

    for i in unconnected_idx:
        node = swc[i, :].squeeze()
        # A recursive search for all the nodes in its subtree, including itself
        subtreenodeids = get_subtree_nodeids(swc, node) 
        nodes2del = np.hstack((nodes2del, subtreenodeids))

    nodes2del = nodes2del.flatten()
    swc = np.asarray([node for node in swc if node[0] not in nodes2del])

    return swc