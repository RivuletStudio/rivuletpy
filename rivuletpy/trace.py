import math
from tqdm import tqdm
import numpy as np
from random import random, randrange
import skfmm
import msfm
from collections import Counter
from scipy.ndimage.measurements import center_of_mass
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage import binary_dilation
from .utils.preprocessing import distgradient
from .utils.swc import getradius, cleanswc, match, match_r1
from filtering.morphology import ssm


class Soma(object):
    def __init__(self, pos, radius, mask=None):
        self.pos = pos
        self.radius = radius
        self.mask = None

    def make_soma_mask(self, bimg):
        '''
        Make soma binary mask with the original
        binary image and its radius and position
        '''

        # Make a ball like mask with 2 X somaradius
        ballvolume = np.zeros(bimg.shape)
        ballvolume[self.pos[0], self.pos[1], self.pos[2]] = 1
        stt = generate_binary_structure(3, 1)
        for i in range(math.ceil(self.radius * 2.5)):
            ballvolume = binary_dilation(ballvolume, structure=stt)

        # Make the soma mask with the intersection
        #between the ball area and the original binary
        self.mask = np.logical_and(ballvolume, bimg)


def r2(img,
       threshold,
       speed='dt',
       is_msfm=True,
       ssmiter=20,
       silence=False,
       clean=False,
       radius=False,
       render=False,
       soma_detection=False,
       somamask=None,
       fast=False):
    '''
    The main entry for rivulet2 tracing algorithm
    Note: the returned swc has 8 columns where the
    8-th column is the online confidence
    '''

    if threshold < 0:
        try:
            from skimage import filters
        except ImportError:
            from skimage import filter as filters
        threshold = filters.threshold_otsu(img)

    if not silence:
        print('--DT to get soma location with threshold:', threshold)
    bimg = (img > threshold).astype('int')  # Segment image
    dt = skfmm.distance(bimg, dx=1.1)  # Boundary DT
    somaradius = dt.max()
    if not silence:
        print('-- Soma radius:', somaradius)
    somapos = np.asarray(np.unravel_index(dt.argmax(), dt.shape))
    marchmap = np.ones(img.shape)
    marchmap[somapos[0], somapos[1], somapos[2]] = -1

    # Make the soma object
    soma = Soma(somapos, somaradius * 2)
    if not silence:
        print('-- Making Soma Mask...')
    soma.make_soma_mask(bimg)

    ## Trace
    if threshold < 0:
        threshold = filters.threshold_otsu(img)
        if not silence:
            print('--Otus for threshold: ', threshold)
    else:
        if not silence:
            print('--Using the user threshold:', threshold)

    img = (img > threshold).astype('int')  # Segment image

    if not silence:
        print('--Boundary DT...')

    dt = skfmm.distance(img, dx=5e-2)  # Boundary DT
    # dtmax = dt.max()
    maxdpt = np.asarray(np.unravel_index(dt.argmax(), dt.shape))
    marchmap = np.ones(img.shape)
    marchmap[maxdpt[0], maxdpt[1], maxdpt[2]] = -1

    somapos = np.asarray(np.unravel_index(dt.argmax(), dt.shape))
    print('Original soma point is', somapos)

    # Old soma position will be replaced by the new soma centroid
    if soma_detection:
        somabimg = (somamask > 0).astype('int')

        # Calculate the new centroid using the soma volume
        newsomapos = center_of_mass(somabimg)

        # Round the float coordinates into integers
        newsomapos = np.round(newsomapos)

        # Release the memory of binary soma image
        del somabimg, somapos
        somapos = newsomapos.astype('int')
        print('The new calculated soma point is', somapos)
    if speed == 'ssm':
        if not silence:
            print('--SSM with GVF...')
        dt = ssm(dt, anisotropic=True, iterations=ssmiter)
        img = dt > filters.threshold_otsu(dt)
        dt = skfmm.distance(img, dx=5e-2)

        if not silence: print('--Reverse DT...')
        dt = skfmm.distance(np.logical_not(dt), dx=5e-3)
        dt[dt > 0.04] = 0.04
        dt = dt.max() - dt

    # # Fast Marching
    if is_msfm:
        if not silence: print('--MSFM...')
        t = msfm.run(makespeed(dt), somapos, False, True)
    else:
        if not silence: print('--FM...')
        t = skfmm.travel_time(marchmap, makespeed(dt), dx=5e-3)

    # Iterative Back Tracking with Erasing
    if not silence: print('--Start Backtracking...')
    swc = iterative_backtrack(
        t,
        img,
        somapos,
        somaradius,
        soma_detection,
        somamask,
        render=render,
        silence=silence,
        eraseratio=1.7 if speed == 'ssm' else 1.5,
        length=5)

    # Clean SWC
    if clean:
        # Only keep the largest connected component of the graph in swc
        print('-- Cleaning swc')
        swc = cleanswc(swc, radius)
    elif not radius:
        swc[:, 5] = 1

    return swc, soma


def makespeed(dt, threshold=0):
    '''
    Make speed image for FM from distance transform
    '''

    F = dt**4
    F[F <= threshold] = 1e-10

    return F


def iterative_backtrack(t,
                        bimg,
                        somapt,
                        somaradius,
                        soma_detection,
                        somamask,
                        length=6,
                        render=False,
                        silence=False,
                        eraseratio=1.1):
    '''
    Trace the segmented image with a single neuron using Rivulet2 algorithm.

    Parameters
    ----------------
    t  :  The time-crossing map generated by fast-marching
    bimg  :  The binary image as 3D numpy ndarray with
    foreground (True) and background (False)
    somapt  :  The soma position as a 3D coordinate in 3D numpy ndarray
    somaradius  :  The approximate soma radius
    render  :  The flag to render the tracing progress for debugging
    silence  :  The flag to silent the tracing progress
    without showing the progress bar
    eraseratio  :  The ratio to enlarge the inital surface of the branch erasing
    '''

    config = {'coverage': 0.98, 'gap': 15}

    # Get the gradient of the Time-crossing map
    dx, dy, dz = distgradient(t.astype('float64'))
    standard_grid = (np.arange(t.shape[0]), np.arange(t.shape[1]),
                     np.arange(t.shape[2]))
    ginterp = (RegularGridInterpolator(standard_grid, dx),
               RegularGridInterpolator(standard_grid, dy),
               RegularGridInterpolator(standard_grid, dz))

    bounds = t.shape
    tt = t.copy()
    tt[bimg <= 0] = -2

    # Label all voxels of soma with -3
    if soma_detection:
        tt[somamask > 0] = -3
        print('Somamask modifies the time map')
    # For making a large tube to contain the last traced branch
    bb = np.zeros(shape=tt.shape)

    if render:
        from .utils.rendering3 import Viewer3, Line3, Ball3
        viewer = Viewer3(800, 800, 800)
        viewer.set_bounds(0, bounds[0], 0, bounds[1], 0, bounds[2])

    # Start tracing loop
    nforeground = bimg.sum()
    coverage = 0.0
    # iteridx = 0
    swc = None
    if not silence:
        pbar = tqdm(total=math.floor(nforeground * config['coverage']))
    velocity = None
    coveredctr_old = 0

    while coverage < config['coverage']:
        coveredctr_new = np.logical_and(tt < 0, bimg > 0).sum()
        coverage = coveredctr_new / nforeground
        if not silence:
            pbar.update(coveredctr_new - coveredctr_old)
        coveredctr_old = coveredctr_new

        # Find the geodesic furthest point on foreground time-crossing-map
        endpt = srcpt = np.asarray(np.unravel_index(tt.argmax(
        ), tt.shape)).astype('float64')

        # Trace it back to maxd
        branch = [srcpt, ]
        branch_conf = [1, ]
        reached = False
        touched = False
        fgctr = 0  # Count how many steps are made on foreground in this branch
        steps_after_reach = 0
        reachedsoma = False
        gapdist = 0  # Length of the branch on gap

        # For online confidence comupting
        online_voxsum = 0.
        low_online_conf = False

        line_color = [random(), random(), random()]

        rlist = []
        while True:  # Start 1 Back-tracking iteration
            try:
                endpt = rk4(srcpt, ginterp, t, 1)
                endptint = [math.floor(p) for p in endpt]
                velocity = endpt - srcpt

                # Count the number of steps it travels on the foreground
                endpt_b = bimg[endptint[0], endptint[1], endptint[2]]
                fgctr += endpt_b

                # Check for the large gap criterion
                if not endpt_b:
                    # Get the mean radius so far
                    rmean = 1 if len(rlist) == 0 else np.mean(rlist)
                    stepsz = np.linalg.norm(srcpt - endpt)
                    gapdist += stepsz

                    if gapdist > rmean * 8:
                        break
                else:
                    gapdist = 0  # Reset gapdist

                # Compute the online confidence
                online_voxsum += endpt_b
                online_confidence = online_voxsum / (len(branch) + 1)

                # Reach somatic area or the distance between the somatic centroid
                # And traced point is less than somatic radius
                if soma_detection:
                    soma_one = (tt[endptint[0], endptint[1], endptint[2]] == -3)
                    soma_criteria = soma_one
                else:
                    # Stop due to reaching soma point
                    soma_criteria = (np.linalg.norm(somapt - endpt) < 1.2 * somaradius)
                
                if soma_criteria:
                    reachedsoma = True
                    if soma_detection:

                        # Initial branch length is set to zero for each branch
                        branchlen = 0

                        for i in range(len(branch)-1):
                            # The starting point 
                            stptbr = branch[i]

                            # Convert it to Numpy for better manipulation
                            np.asarray(stptbr)

                            # The point adjacent to the starting point
                            ntptbr = branch[i+1]

                            # Convert it to numpy for better manipulation
                            np.asarray(ntptbr)

                            # Use Numpy subtract rather than minus
                            # operation sign to avoid dimension confusion
                            diffpt = np.subtract(stptbr, ntptbr)

                            # The variable normbt is the step length for each step
                            normnt = np.linalg.norm(diffpt)

                            # Calculate the length of whole branch iteratively
                            branchlen = branchlen + normnt

                        # Only the long branches connected to the soma are preserved.
                        if (branchlen < 15):
                            low_online_conf = True

                    # Render a yellow node at fork point
                    if render:
                        ball = Ball3((endpt[0], endpt[1], endpt[2]), radius=1)
                        ball.set_color(0.917, 0.933, 0.227)
                        viewer.add_geom(ball)
                    break

                # Render the line segment
                if render:
                    l = Line3(srcpt, endpt)
                    l.set_color(*line_color)
                    viewer.add_geom(l)
                    viewer.render(return_rgb_array=False)

                # Consider reaches previous explored area traced with branch
                # Note: when the area was traced due to noise points
                # (erased with -2), not considered as 'reached'
                if tt[endptint[0], endptint[1], endptint[2]] == -1:
                    reached = True

                # If the endpoint reached previously traced area check for
                # node to connect for at each step
                if reached:
                    if swc is None:
                        break  # There has not been any branch added yet

                    steps_after_reach += 1
                    endradius = getradius(bimg, endpt[0], endpt[1], endpt[2])
                    touched, touchidx = match(swc, endpt, endradius)
                    # closestnode = swc[touchidx, :]

                    if touched or steps_after_reach >= 100:
                        # Render a blue node at fork point
                        if touched and render:
                            ball = Ball3(
                                (endpt[0], endpt[1], endpt[2]), radius=1)
                            ball.set_color(0, 0, 1)
                            viewer.add_geom(ball)
                        break

                # If the velocity is too small, sprint a bit with the momentum
                if np.linalg.norm(velocity) <= 0.5 and len(branch) >= length:
                    endpt = srcpt + (branch[-1] - branch[-4])

                if len(branch) > 15 and np.linalg.norm(branch[-15] -
                                                       endpt) < 1.:
                    # Render a brown node at stopping point since not moving
                    if render:
                        ball = Ball3((endpt[0], endpt[1], endpt[2]), radius=1)
                        ball.set_color(0.729, 0.192, 0.109)
                        viewer.add_geom(ball)
                    break  # There could be zero gradients somewhere

                if online_confidence < 0.25:
                    low_online_conf = True

                    # Render a grey node at stopping point with low confidence
                    if render:
                        ball = Ball3((endpt[0], endpt[1], endpt[2]), radius=1)
                        ball.set_color(0.5, 0.5, 0.5)
                        viewer.add_geom(ball)
                    break

                # All in vain finally if it traces out of bound
                if not inbound(endpt, tt.shape):
                    break

            except ValueError:
                # Render a pink node at value error
                if render:
                    ball = Ball3((endpt[0], endpt[1], endpt[2]), radius=1)
                    ball.set_color(0.972, 0.607, 0.619)
                    viewer.add_geom(ball)
                break

            branch.append(endpt)  # Add the newly traced node to current branch
            r = getradius(bimg, endpt[0], endpt[1], endpt[2])
            rlist.append(r)
            branch_conf.append(online_confidence)
            srcpt = endpt  # Shift forward

        # Check forward confidence
        cf = conf_forward(branch, bimg)

        ## Erase it from the timemap
        for node in branch:
            n = [math.floor(n) for n in node]
            r = getradius(bimg, n[0], n[1], n[2])
            r = 1 if r < 1 else r
            rlist.append(r)

            # To make sure all the foreground voxels are included in bb
            r *= eraseratio
            r = math.ceil(r)
            X, Y, Z = np.meshgrid(
                constrain_range(n[0] - r, n[0] + r + 1, 0, tt.shape[0]),
                constrain_range(n[1] - r, n[1] + r + 1, 0, tt.shape[1]),
                constrain_range(n[2] - r, n[2] + r + 1, 0, tt.shape[2]))
            bb[X, Y, Z] = 1

        startidx = [math.floor(p) for p in branch[0]]
        endidx = [math.floor(p) for p in branch[-1]]

        if len(branch) > length and tt[endidx[0], endidx[1], endidx[2]] < tt[
                startidx[0], startidx[1], startidx[2]]:
            erase_region = np.logical_and(
                tt[endidx[0], endidx[1], endidx[2]] <= tt,
                tt <= tt[startidx[0], startidx[1], startidx[2]])
            erase_region = np.logical_and(bb, erase_region)
        else:
            erase_region = bb.astype('bool')

        if np.count_nonzero(erase_region) > 0:
            tt[erase_region] = -2 if low_online_conf else -1
        bb.fill(0)

        if touched:
            connectid = swc[touchidx, 0]
        elif reachedsoma:
            connectid = 0
        else:
            connectid = None

        # Check the confidence of this branch
        if cf[-1] < 0.5 or low_online_conf:
            continue

        for i, node in enumerate(branch):
            n = [math.floor(n) for n in node]
            if tt[n[0], n[1], n[2]] == -2:
                branch_conf[i] = 0

        swc = add2swc(swc, branch, rlist, branch_conf, connectid)

    # After all tracing iterations, check all unconnected nodes
    for nodeidx in range(swc.shape[0]):
        if swc[nodeidx, 6] == -2:
            # Find the closest node in swc, excluding the nodes traced earlier than this node in match
            swc2consider = swc[swc[:, 0] > swc[nodeidx, 0], :]
            connect, minidx = match(swc2consider, swc[nodeidx, 2:5], 3)
            if connect:
                swc[nodeidx, 6] = swc2consider[minidx, 0]

    # Prune short leaves 
    swc = prune_leaves(swc, bimg, length, 0.5)

    # Add soma node to the result swc
    somanode = np.asarray(
        [0, 1, somapt[0], somapt[1], somapt[2], somaradius, -1, 1.])
    swc = np.vstack((somanode, swc))
    if not silence:
        pbar.close()  # Close the progress bar

    return swc  # The real swc and the confidence array


def iterative_backtrack_r1(t,
                           bimg,
                           somapt,
                           somaradius,
                           gap=8,
                           wiring=1.5,
                           length=4,
                           render=False,
                           silence=True):
    '''
    Trace the segmented image with a single neuron using Rivulet1 algorithm.
    [1] Liu, Siqi, et al. "Rivulet: 3D Neuron Morphology Tracing with Iterative Back-Tracking." Neuroinformatics (2016): 1-15.
    [2] Zhang, Donghao, et al. "Reconstruction of 3D neuron morphology using Rivulet back-tracking." 
    Biomedical Imaging (ISBI), 2016 IEEE 13th International Symposium on. IEEE, 2016.

    This algorithm is deprecated from the standard Rivulet pipeline since Rivulet2 is more accurate and faster than Rivulet1.
    This routine is kept for algorithmic experiments to see the difference between two version of Rivulet
    '''

    config = {'coverage': 0.98}

    # Get the gradient of the Time-crossing map
    dx, dy, dz = distgradient(t.astype('float64'))
    standard_grid = (np.arange(t.shape[0]), np.arange(t.shape[1]),
                     np.arange(t.shape[2]))
    ginterp = (RegularGridInterpolator(standard_grid, dx),
               RegularGridInterpolator(standard_grid, dy),
               RegularGridInterpolator(standard_grid, dz))

    bounds = t.shape
    tt = t.copy()
    tt[bimg == 0] = -2
    bb = np.zeros(
        shape=tt.
        shape)  # For making a large tube to contain the last traced branch

    if render:
        from .utils.rendering3 import Viewer3, Line3, Ball3
        viewer = Viewer3(800, 800, 800)
        viewer.set_bounds(0, bounds[0], 0, bounds[1], 0, bounds[2])

    # Start tracing loop
    nforeground = bimg.sum()
    coverage = 0.0
    iteridx = 0
    swc = None
    if not silence:
        pbar = tqdm(total=math.floor(nforeground * config['coverage']))
    velocity = None
    coveredctr_old = 0

    while coverage < config['coverage']:
        iteridx += 1
        coveredctr_new = np.logical_and(tt < 0, bimg > 0).sum()
        coverage = coveredctr_new / nforeground
        if not silence: pbar.update(coveredctr_new - coveredctr_old)
        coveredctr_old = coveredctr_new

        # Find the geodesic furthest point on foreground time-crossing-map
        endpt = srcpt = np.asarray(np.unravel_index(tt.argmax(
        ), tt.shape)).astype('float64')

        # Trace it back to maxd 
        branch = [srcpt, ]
        reached = False
        touched = False
        notmoving = False
        valueerror = False
        # gapctr = 0 # Count continous steps on background
        gapdist = 0.  # The distance of the gap measured in voxel space
        # steps_after_reach = 0
        outofbound = reachedsoma = False
        line_color = [random(), random(), random()]

        while True:  # Start 1 Back-tracking iteration
            try:
                endpt = rk4(srcpt, ginterp, t, 1)
                endptint = [math.floor(p) for p in endpt]
                endptint_ceil = [math.ceil(p) for p in endpt]
                velocity = endpt - srcpt
                velnorm = np.linalg.norm(velocity)

                # See if it travels too far on the background
                endpt_b = bimg[endptint[0], endptint[1], endptint[2]] or bimg[
                    endptint_ceil[0], endptint_ceil[1], endptint_ceil[2]]
                # print('')
                # gapctr = 0 if endpt_b else gapctr + 1
                gapdist = 0 if endpt_b > 0 else gapdist + velnorm
                # if gapdist > 0:
                #     print('gap:', gapdist)
                if gapdist > gap:
                    break  # Stop tracing if gap is too big

                if np.linalg.norm(
                        somapt - endpt
                ) < 1.2 * somaradius:  # Stop due to reaching soma point
                    reachedsoma = True

                    # Render a yellow node at fork point
                    if render:
                        ball = Ball3((endpt[0], endpt[1], endpt[2]), radius=1)
                        ball.set_color(0.917, 0.933, 0.227)
                        viewer.add_geom(ball)
                    break

                # Render the line segment
                if render:
                    l = Line3(srcpt, endpt)
                    l.set_color(*line_color)
                    viewer.add_geom(l)
                    viewer.render(return_rgb_array=False)

                # Consider reaches previous explored area traced with real branch
                # Note: when the area was traced due to noise points (erased with -2), not considered as 'reached'
                if tt[endptint[0], endptint[1], endptint[2]] == -1:
                    reached = True

                if reached:  # If the endpoint reached previously traced area check for node to connect for at each step
                    if swc is None:
                        break  # There has not been any branch added yet
                    endradius = getradius(bimg, endpt[0], endpt[1], endpt[2])
                    touched, touchidx = match_r1(swc, endpt, endradius, wiring)
                    closestnode = swc[touchidx, :]

                    if touched and render:  # Render a blue node at fork point
                        ball = Ball3((endpt[0], endpt[1], endpt[2]), radius=1)
                        ball.set_color(0, 0, 1)
                        viewer.add_geom(ball)
                    break

                # # If the velocity is too small, sprint a bit with the momentum
                if velnorm <= 0.5 and len(branch) >= length:
                    endpt = srcpt + (branch[-1] - branch[-4])

                if len(branch) > 15 and np.linalg.norm(branch[-15] -
                                                       endpt) < 1.:
                    notmoving = True
                    # Render a brown node at stopping point since not moving
                    if render:
                        ball = Ball3((endpt[0], endpt[1], endpt[2]), radius=1)
                        ball.set_color(0.729, 0.192, 0.109)
                        viewer.add_geom(ball)

                    break  # There could be zero gradients somewhere

                # All in vain finally if it traces out of bound
                if not inbound(endpt, tt.shape):
                    outofbound = True
                    break

            except ValueError:
                valueerror = True
                # Render a pink node at value error 
                if render:
                    ball = Ball3((endpt[0], endpt[1], endpt[2]), radius=1)
                    ball.set_color(0.972, 0.607, 0.619)
                    viewer.add_geom(ball)
                break

            branch.append(endpt)  # Add the newly traced node to current branch
            srcpt = endpt  # Shift forward

        ## Erase it from the timemap
        rlist = []
        for node in branch:
            n = [math.floor(n) for n in node]
            r = getradius(bimg, n[0], n[1], n[2])
            r = 1 if r < 1 else r
            rlist.append(r)

            # To make sure all the foreground voxels are included in bb
            r *= 0.8
            r = math.ceil(r)
            X, Y, Z = np.meshgrid(
                constrain_range(n[0] - r, n[0] + r + 1, 0, tt.shape[0]),
                constrain_range(n[1] - r, n[1] + r + 1, 0, tt.shape[1]),
                constrain_range(n[2] - r, n[2] + r + 1, 0, tt.shape[2]))
            bb[X, Y, Z] = 1

        erase_region = bb.astype('bool')
        if np.count_nonzero(erase_region) > 0:
            tt[erase_region] = -1
        bb.fill(0)

        if touched:
            connectid = swc[touchidx, 0]
        elif reachedsoma:
            connectid = 0
        else:
            connectid = None

        # Dump due to low confidence
        # cf = conf_vox(branch, bimg)
        # if cf < 0.1:
        #     continue

        swc = add2swc(swc, branch, rlist, connectid)
        # if notmoving: swc[-1, 1] = 128 # Some weired colour for unexpected stop
        # if valueerror: swc[-1, 1] = 256 # Some weired colour for unexpected stop

    # After all tracing iterations, check all unconnected nodes
    for nodeidx in range(swc.shape[0]):
        if swc[nodeidx, 6] == -2:
            # Find the closest node in swc, excluding the nodes traced earlier than this node in match
            swc2consider = swc[swc[:, 0] > swc[nodeidx, 0], :]
            connect, minidx = match_r1(swc2consider, swc[nodeidx, 2:5], 3,
                                       wiring)
            if connect: swc[nodeidx, 6] = swc2consider[minidx, 0]

    # Prune short leaves 
    swc = prune_leaves(swc, bimg, length, 0.)

    # Add soma node to the result swc
    somanode = np.asarray(
        [0, 1, somapt[0], somapt[1], somapt[2], somaradius, -1])
    swc = np.vstack((somanode, swc))

    return swc


def conf_vox(branch, bimg):
    '''
    The confidence score used in Rivulet1. 
        The propotion of foreground voxels on a branch. Repeatant voxels will only be counted once

    Parameters
    ----------------
    branch: list of 1 X 3 np.ndarray 
    bimg: the binary image (3D np.ndarray)
    '''
    voxhash = {}
    for node in branch:
        nodevox = tuple([math.floor(x) for x in node])
        voxhash[nodevox] = bimg[nodevox[0], nodevox[1], nodevox[2]]

    foresum = np.sum(np.asarray(list(voxhash.values())))
    return foresum / len(voxhash)


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
    k1 *= stepsize / max(np.linalg.norm(k1), 1.)
    tp = srcpt - 0.5 * k1  # Position of temporary point
    if not inbound(tp, t.shape):
        return srcpt

    # Compute K2
    k2 = np.asarray([g(tp)[0] for g in ginterp])
    k2 *= stepsize / max(np.linalg.norm(k2), 1.)
    tp = srcpt - 0.5 * k2  # Position of temporary point
    if not inbound(tp, t.shape):
        return srcpt

    # Compute K3
    k3 = np.asarray([g(tp)[0] for g in ginterp])
    k3 *= stepsize / max(np.linalg.norm(k3), 1.)
    tp = srcpt - k3  # Position of temporary point
    if not inbound(tp, t.shape):
        return srcpt

    # Compute K4
    k4 = np.asarray([g(tp)[0] for g in ginterp])
    k4 *= stepsize / max(np.linalg.norm(k4), 1.)

    return srcpt - (k1 + k2 * 2 + k3 * 2 + k4) / 6.0  # Compute final point


def inbound(pt, shape):
    return all([True if 0 <= p <= s - 1 else False for p, s in zip(pt, shape)])


def fibonacci_sphere(samples=1, randomize=True):
    rnd = 1.
    if randomize:
        rnd = random() * samples

    points = []
    offset = 2. / samples
    increment = math.pi * (3. - math.sqrt(5.))

    for i in range(samples):
        y = ((i * offset) - 1) + (offset / 2)
        r = math.sqrt(1 - pow(y, 2))

        phi = ((i + rnd) % samples) * increment

        x = math.cos(phi) * r
        z = math.sin(phi) * r

        points.append(np.array([x, y, z]))

    return points


def add2swc(swc, path, radius, branch_conf, connectid=None, random_color=True):
    '''
    Add a branch to swc.
    Note: This swc is special with N X 8 shape. The 8-th column is the online confidence
    '''

    if random_color:
        rand_node_type = randrange(256)

    newbranch = np.zeros((len(path), 7))
    if swc is None:  # It is the first branch to be added
        idstart = 1
    else:
        idstart = swc[:, 0].max() + 1

    for i, p in enumerate(path):
        id = idstart + i
        nodetype = 3  # 3 for basal dendrite; 4 for apical dendrite; However now we cannot differentiate them automatically

        if i == len(path) - 1:  # The end of this branch
            pid = -2 if connectid is None else connectid
            if connectid is not None and connectid is not 1 and swc is not None:
                swc[swc[:, 0] == connectid,
                    1] = 5  # its connected node is fork point
        else:
            pid = idstart + i + 1
            if i == 0:
                nodetype = 6  # Endpoint

        newbranch[i] = np.asarray([
            id, rand_node_type
            if random_color else nodetype, p[0], p[1], p[2], radius[i], pid
        ])

    branch_conf = np.reshape(np.asarray(branch_conf), (len(branch_conf), 1))
    newbranch = np.hstack((newbranch, branch_conf))

    if swc is None:
        swc = newbranch
    else:
        # Check if any tail should be connected to its head
        head = newbranch[0]
        matched, minidx = match(swc, head[2:5], head[5])
        if matched and swc[minidx, 6] is -2: swc[minidx, 6] = head[0]
        swc = np.vstack((swc, newbranch))

    return swc


def constrain_range(min, max, minlimit, maxlimit):
    return list(
        range(min if min > minlimit else minlimit, max
              if max < maxlimit else maxlimit))


def prune_leaves(swc, img, length, conf):

    # Find all the leaves
    childctr = Counter(swc[:, 6])
    leafidlist = [id for id in swc[:, 0]
                  if id not in swc[:, 6]]  # Does not work
    id2dump = []

    for leafid in leafidlist:  # Iterate each leaf node
        nodeid = leafid
        branch = []
        while True:  # Get the leaf branch out
            node = swc[swc[:, 0] == nodeid, :].flatten()
            if node.size == 0:
                break
            branch.append(node)
            parentid = node[6]
            if childctr[parentid] is not 1:
                break  # merged / unconnected
            nodeid = parentid

        # Prune if the leave is too short or
        # the confidence of the leave branch is too low
        if len(branch) < length or conf_forward([b[2:5] for b in branch],
                                                img)[-1] < conf:
            id2dump.extend([node[0] for node in branch])

    # Only keep the swc nodes not in the dump id list
    cuttedswc = []
    for nodeidx in range(swc.shape[0]):
        if swc[nodeidx, 0] not in id2dump:
            cuttedswc.append(swc[nodeidx, :])

    cuttedswc = np.squeeze(np.dstack(cuttedswc)).T
    return cuttedswc


def conf_forward(path, img):
    conf_forward = np.zeros(shape=(len(path), ))
    branchvox = np.asarray([
        img[math.floor(p[0]), math.floor(p[1]), math.floor(p[2])] for p in path
    ])
    for i in range(len(path, )):
        conf_forward[i] = branchvox[:i].sum() / (i + 1)

    return conf_forward
