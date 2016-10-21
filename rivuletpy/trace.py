import os
import numpy as np
from scipy import ndimage 
from .utils.backtrack import *
from .utils.preprocessing import distgradient
import progressbar
from scipy.interpolate import RegularGridInterpolator 
from random import random
from skimage.morphology import skeletonize_3d
import skfmm

def makespeed(dt, threshold=0):
    F = dt ** 4
    F[F<=threshold] = 1e-10
    return F


def iterative_backtrack(t, bimg, somapt, somaradius, somaimg, render=False, silence=False, eraseratio=1.1):
    '''Trace the 3d tif with a single neuron using Rivulet algorithm'''
    config = {'length':6, 'coverage':0.98, 'gap':15}

    # Get the gradient of the Time-crossing map
    dx, dy, dz = distgradient(t.astype('float64'))
    standard_grid = (np.arange(t.shape[0]), np.arange(t.shape[1]), np.arange(t.shape[2]))
    ginterp = (RegularGridInterpolator(standard_grid, dx),
               RegularGridInterpolator(standard_grid, dy),
               RegularGridInterpolator(standard_grid, dz))

    bounds = t.shape
    tt = t.copy()
    tt[bimg <= 0] = -2

    # Label all voxels of soma with -3
    tt[somaimg > 0] = -3
    bb = np.zeros(shape=tt.shape) # For making a large tube to contain the last traced branch

    if render:
        from .utils.rendering3 import Viewer3, Line3, Ball3
        viewer = Viewer3(800, 800, 800)
        viewer.set_bounds(0, bounds[0], 0, bounds[1], 0, bounds[2])

    # Start tracing loop
    nforeground = bimg.sum()
    converage = 0.0
    iteridx = 0
    swc = None
    if not silence: bar = progressbar.ProgressBar(max_value=nforeground)
    velocity = None

    while converage < config['coverage']:
        iteridx += 1
        coveredctr = np.logical_and(tt<0, bimg > 0).sum() 
        converage =  coveredctr / nforeground

        # Find the geodesic furthest point on foreground time-crossing-map
        endpt = srcpt = np.asarray(np.unravel_index(tt.argmax(), tt.shape)).astype('float64')
        if not silence: bar.update(coveredctr)

        # Trace it back to maxd 
        branch = [srcpt,]
        reached = False
        touched = False
        notmoving =False 
        valueerror = False 
        gapctr = 0 # Count continous steps on background
        fgctr = 0 # Count how many steps are made on foreground in this branch
        steps_after_reach = 0
        outofbound = reachedsoma = False

        # For online confidence comupting
        online_voxsum = 0.
        low_online_conf = False

        line_color = [random(), random(), random()]

        while True: # Start 1 Back-tracking iteration
            try:
                endpt = rk4(srcpt, ginterp, t, 1)
                endptint = [math.floor(p) for p in endpt]
                velocity = endpt - srcpt

                # See if it travels too far on the background
                endpt_b = bimg[endptint[0], endptint[1], endptint[2]]
                gapctr = 0 if endpt_b else gapctr + 1
                fgctr += endpt_b

                # Compute the online confidence
                online_voxsum += endpt_b
                online_confidence = online_voxsum / (len(branch) + 1)

                # Reach somatic area or the distance between the somatic centroid
                # And traced point is less than somatic radius
                if (tt[endptint[0], endptint[1], endptint[2]] == -3) | \
                 (np.linalg.norm(somapt - endpt) < somaradius):
                # if np.linalg.norm(somapt - endpt) < 1.2 * somaradius: # Stop due to reaching soma point
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

                if reached: # If the endpoint reached previously traced area check for node to connect for at each step
                    if swc is None: break # There has not been any branch added yet

                    steps_after_reach += 1
                    endradius = getradius(bimg, endpt[0], endpt[1], endpt[2])
                    touched, touchidx = match(swc, endpt, endradius)
                    closestnode = swc[touchidx, :]

                    if touched or steps_after_reach >= 100: 
                        # Render a blue node at fork point
                        if touched and render:
                            ball = Ball3((endpt[0], endpt[1], endpt[2]), radius=1)
                            ball.set_color(0, 0, 1)
                            viewer.add_geom(ball)
                        break

                # If the velocity is too small, sprint a bit with the momentum
                if np.linalg.norm(velocity) <= 0.5 and len(branch) >= config['length']:
                    endpt = srcpt + (branch[-1] - branch[-4])

                if len(branch) > 15 and np.linalg.norm(branch[-15] - endpt) < 1.: 
                    notmoving = True
                    # print('==Not Moving - Velocity:', velocity)
                    # Render a brown node at stopping point since not moving
                    if render:
                        ball = Ball3((endpt[0], endpt[1], endpt[2]), radius=1)
                        ball.set_color(0.729, 0.192, 0.109)
                        viewer.add_geom(ball)
                    break # There could be zero gradients somewhere

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
                    outofbound = True
                    break

            except ValueError:
                valueerror = True
                print('== Value ERR - Velocity:', velocity, 'Point:', endpt)
                # Render a pink node at value error 
                if render:
                    ball = Ball3((endpt[0], endpt[1], endpt[2]), radius=1)
                    ball.set_color(0.972, 0.607, 0.619)
                    viewer.add_geom(ball)
                break 

            branch.append(endpt) # Add the newly traced node to current branch
            srcpt = endpt # Shift forward

        # Check forward confidence 
        cf = conf_forward(branch, bimg)

        ## Erase it from the timemap
        rlist = []
        for node in branch:
            n = [math.floor(n) for n in node]
            r = getradius(bimg, n[0], n[1], n[2])
            r = 1 if r < 1 else r
            rlist.append(r)
            
            # To make sure all the foreground voxels are included in bb
            r *= eraseratio
            r = math.ceil(r)
            X, Y, Z = np.meshgrid(constrain_range(n[0]-r, n[0]+r+1, 0, tt.shape[0]),
                                  constrain_range(n[1]-r, n[1]+r+1, 0, tt.shape[1]),
                                  constrain_range(n[2]-r, n[2]+r+1, 0, tt.shape[2]))
            bb[X, Y, Z] = 1

        startidx = [math.floor(p) for p in branch[0]]
        endidx = [math.floor(p) for p in branch[-1]]

        if len(branch) > config['length'] and tt[endidx[0], endidx[1], endidx[2]] < tt[startidx[0], startidx[1], startidx[2]]:
            erase_region = np.logical_and(tt[endidx[0], endidx[1], endidx[2]] <= tt, tt <= tt[startidx[0], startidx[1], startidx[2]])
            erase_region = np.logical_and(bb, erase_region)
        else:
            erase_region = bb.astype('bool')

        if np.count_nonzero(erase_region) > 0:
            tt[erase_region] = -2 if low_online_conf else -1
        bb.fill(0)
            
        # if len(branch) > config['length']: 
        if touched:
            connectid = swc[touchidx, 0]
        elif reachedsoma:
            connectid = 0 
        else:
            connectid = None

        if cf[-1] < 0.5 or low_online_conf: # Check the confidence of this branch
            continue 

        swc = add2swc(swc, branch, rlist, connectid)
        if notmoving: swc[-1, 1] = 128 # Some weired colour for unexpected stop
        if valueerror: swc[-1, 1] = 256 # Some weired colour for unexpected stop

    # After all tracing iterations, check all unconnected nodes
    for nodeidx in range(swc.shape[0]):
        if swc[nodeidx, -1]  == -2:
            # Find the closest node in swc, excluding the nodes traced earlier than this node in match
            swc2consider = swc[swc[:, 0] > swc[nodeidx, 0], :]
            connect, minidx = match(swc2consider, 
                                                     swc[nodeidx, 2:5], 3)
            if connect:
                swc[nodeidx, -1] = swc2consider[minidx, 0]
            # else:
            #     swc[nodeidx, 1] = 200 

    # Prune short leaves 
    swc = prune_leaves(swc, bimg, config['length'], 0.5)

    # Add soma node to the result swc
    somanode = np.asarray([0, 1, somapt[0], somapt[1], somapt[2], somaradius, -1])
    swc = np.vstack((somanode, swc))

    return swc


