import os
import numpy as np
from scipy import ndimage 
from libtiff import TIFF 
from .utils.io import *
from .utils.preprocessing import *
from .utils.backtrack import *
from .utils.rendering3 import *
from matplotlib import pyplot as plt

import progressbar

def trace(img, **userconfig):
    '''Trace the 3d tif with a single neuron using Rivulet algorithm'''
    config = {'length':5,
              'coverage':0.98,
              'threshold':0,
              'gap':15, 
              'ignore_radius': False,
              'render':False, 
              'toswcfile':None, 
              'silence':False,
              'skedt': False, # True if the distance transform is generated with skelontonization algorithm
              'clean': False}
    config.update(userconfig)
    # if not config['silence']: print('Start Tracing', filepath)

    dt, t, ginterp, bimg, cropregion = rivulet_preprocessing(img, config)

    dtmax = dt.max()
    maxdpt = np.asarray(np.unravel_index(dt.argmax(), dt.shape))
    print('Image size after crop:', bimg.shape)

    tt = t.copy()
    tt[bimg <= 0] = -2
    bb = np.zeros(shape=tt.shape) # For making a large tube to contain the last traced branch
    forevoxsum = bimg.sum()

    bounds = dt.shape
    if config['render']:
        viewer = Viewer3(800, 800, 800)
        viewer.set_bounds(0, bounds[0], 0, bounds[1], 0, bounds[2])

    idx = np.where(bimg > 0)

    # Start tracing loop
    nforeground = bimg.sum()
    covermap = np.zeros(bimg.shape) 
    converage = 0.0
    iteridx = 0
    swc = None
    if not config['silence']: bar = progressbar.ProgressBar(max_value=1.)
    velocity = None

    while converage < config['coverage']:
        iteridx += 1
        converage = np.logical_and(tt==-1, bimg > 0).sum() / nforeground

        # Find the geodesic furthest point on foreground time-crossing-map
        endpt = srcpt = np.asarray(np.unravel_index(tt.argmax(), tt.shape)).astype('float64')
        # tt[math.floor(endpt[0]), math.floor(endpt[1]), math.floor(endpt[2])] = -1
        if not config['silence']: bar.update(converage)

        # Trace it back to maxd 
        path = [srcpt,]
        reached = False
        touched = False
        gapctr = 0 # Count continous steps on background
        fgctr = 0 # Count how many steps are made on foreground in this branch
        steps_after_reach = 0
        outofbound = reachedsoma = False
        while True:
            try:
                endpt = rk4(srcpt, ginterp, t, 1)
                endptint = [math.floor(p) for p in endpt]
                velocity = endpt - srcpt

                # See if it travels too far on the background
                endpt_b = bimg[endptint[0], endptint[1], endptint[2]]
                gapctr = 0 if endpt_b else gapctr + 1
                fgctr += endpt_b

                if gapctr > config['gap']: 
                    # print('==Stop due to gap at', endpt)
                    break 

                if np.linalg.norm(maxdpt - endpt) < 1.5 * dtmax:
                    reachedsoma = True
                    break

                # Render the line segment
                if config['render']:
                    l = Line3(srcpt, endpt)
                    l.set_color(1., 0., 0)
                    viewer.add_geom(l)
                    viewer.render(return_rgb_array=False)

                if not inbound(endpt, tt.shape): 
                    # print('==Stop due to out of bound at', endpt)
                    outofbound = True
                    break;
                if tt[endptint[0], endptint[1], endptint[2]] == -1:
                    reached = True

                if reached: # If the endpoint reached previously traced area check for node to connect for at each step
                    if swc is None:
                        break;

                    steps_after_reach += 1
                    endradius = getradius(bimg, endpt[0], endpt[1], endpt[2])
                    touched, touchidx = match(swc, endpt, endradius)
                    closestnode = swc[touchidx, :]
                    if touched and config['render']:
                        ball = Ball3((endpt[0], endpt[1], endpt[2]), radius=1)
                        if len(path) < config['length']:
                            ball.set_color(1, 1, 1)
                        else:
                            ball.set_color(0, 0, 1)
                        viewer.add_geom(ball)
                    if touched or steps_after_reach >= 30: break

                if len(path) > 15 and np.linalg.norm(path[-15] - endpt) < 1.:
                    # print('== Stop due to not moving at ', endpt)
                    ttblock = tt[274-2:274+3, 320-2:320+3, 97-2:97+3]
                    # print(ttblock)
                    break;
            except ValueError:
                # print('==ValueError at:', endpt)
                if velocity is not None:
                    endpt = srcpt + velocity
                break

            path.append(endpt)
            srcpt = endpt

        rlist = []
        # Erase it from the timemap
        for node in path:
            n = [math.floor(n) for n in node]
            r = getradius(bimg, n[0], n[1], n[2])
            rlist.append(r)
            
            # To make sure all the foreground voxels are included in bb
            r *= 1.5 if len(path) > config['length'] else 2
            r = math.ceil(r)
            X, Y, Z = np.meshgrid(constrain_range(n[0]-r, n[0]+r+1, 0, tt.shape[0]),
                                  constrain_range(n[1]-r, n[1]+r+1, 0, tt.shape[1]),
                                  constrain_range(n[2]-r, n[2]+r+1, 0, tt.shape[2]))
            bb[X, Y, Z] = 1

        startidx = [math.floor(p) for p in path[0]]
        endidx = [math.floor(p) for p in path[-1]]

        if len(path) > config['length']:
            erase_region = np.logical_and(tt[endidx[0], endidx[1], endidx[2]] <= tt, tt <= tt[startidx[0], startidx[1], startidx[2]])
            erase_region = np.logical_and(bb, erase_region)
        else:
            erase_region = bb.astype('bool')

        if np.count_nonzero(erase_region) > 0:
            tt[erase_region] = -1
        bb.fill(0)
            
        if len(path) > config['length'] and fgctr / len(path) > .5:
            if touched:
                connectid = swc[touchidx, 0]
            elif reachedsoma:
                connectid = -1 
            else:
                connectid = None

            swc = add2swc(swc, path, rlist, connectid)

    if config['clean']: swc = cleanswc(swc) # This will find the nodes with -2 as parents and clean its branch

    if config['toswcfile'] is not None:
        swc[:, 2] += cropregion[0, 0]
        swc[:, 3] += cropregion[1, 0]
        swc[:, 4] += cropregion[2, 0]
        swc_x = swc[:, 2].copy()
        swc_y = swc[:, 3].copy()
        swc[:, 2] = swc_y
        swc[:, 3] = swc_x
        if config['ignore_radius']:
            swc[:,5] = 1
        saveswc(config['toswcfile'], swc)
