import os
import numpy as np
from scipy import ndimage 
from .utils.preprocessing import *
from .utils.backtrack import *
import progressbar

from skimage.morphology import skeletonize_3d
import skfmm

def rivulet_preprocessing(img, config):
    bimg = (img > config['threshold']).astype('int')

    # Distance transform from the background
    if not config['silence']: print('Distance Transform...')

    # The boundary DT
    # The boundary DT is performed with a segmenation on the original image rather than the filtered one
    # if an original image is given
    if config['original_image'] is not None and config['soma_threshold'] is not None:
        obimg = config['original_image'] > config['soma_threshold']
        dt = skfmm.distance(obimg, dx=5e-2)
        dt[obimg==0] = 0
    else:
        dt = skfmm.distance(bimg, dx=5e-2)
        dt[bimg==0] = 0

    dtmax = dt.max()
    marchmap = np.ones(bimg.shape)
    maxdpt = np.asarray(np.unravel_index(dt.argmax(), dt.shape))
    marchmap[maxdpt[0], maxdpt[1], maxdpt[2]] = -1

    if config['response_as_speed']:
        if config['skedt']:
            if not config['silence']: print('Using skelonisation DT...')
            ske = skeletonize_3d(bimg)
            dt = skfmm.distance(np.logical_not(ske), dx=5e-3)
            dt[dt > 0.04] = 0.04
            bimg = dt < 0.02
            # bimg = np.logical_or(bimg, obimg) # Add the soma segmentation as well
            dt = dt.max() - dt

        # Fast marching from the position with the largest distance
        if not config['silence']: print('Fast Marching...')
        F = dt ** 4
        F[F==0] = 1e-10
    else:
        F = img 
        F[F <= config['threshold']] = 1e-10

    t = skfmm.travel_time(marchmap, F, dx=5e-3)
    
    # Get the gradient volume of the time crossing map
    if not config['silence']: print('Getting gradients...')
    gshape = list(t.shape)
    gshape.append(3)
    g = np.zeros(gshape)
    standard_grid = (np.arange(t.shape[0]), np.arange(t.shape[1]), np.arange(t.shape[2]))
    dx, dy, dz = distgradient(t.astype('float64'))
    ginterp = (RegularGridInterpolator(standard_grid, dx),
               RegularGridInterpolator(standard_grid, dy),
               RegularGridInterpolator(standard_grid, dz))

    return dt, t, ginterp, bimg, dtmax, maxdpt # The dtmax and maxdpt are derived from the boundary dt, however dt is the skelonton dt


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

    dt, t, ginterp, bimg, dtmax, maxdpt  = rivulet_preprocessing(img, config)

    # dtmax = dt.max()
    # maxdpt = np.asarray(np.unravel_index(dt.argmax(), dt.shape))
    print('Image size after crop:', bimg.shape)

    tt = t.copy()
    tt[bimg <= 0] = -2
    bb = np.zeros(shape=tt.shape) # For making a large tube to contain the last traced branch
    forevoxsum = bimg.sum()

    bounds = dt.shape
    if config['render']:
        from .utils.rendering3 import Viewer3, Line3, Ball3
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

    if config['clean']:
        # This will only keep the largest connected component of the graph in swc
        print('Cleaning swc')
        swc = cleanswc(swc, not config['ignore_radius']) 

    if not config['clean'] and config['ignore_radius']:
        swc[:, 5] = 1
    
    return swc
