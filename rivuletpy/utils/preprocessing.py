import numpy as np
from .io import *
from scipy.interpolate import RegularGridInterpolator
import skfmm

def crop(img, thr):
    """Crop a 3D block with value > thr"""
    ind = np.argwhere(img > thr)
    x = ind[:,0]
    y = ind[:,1]
    z = ind[:,2]
    xmin = max(x.min()-10, 0)
    xmax = min(x.max()+10, img.shape[0])
    ymin = max(y.min()-10, 1)
    ymax = min(y.max()+10, img.shape[1])
    zmin = max(z.min()-10, 2)
    zmax = min(z.max()+10, img.shape[2])
    
    return img[xmin : xmax,
               ymin : ymax, 
               zmin : zmax], np.array([[xmin, xmax], 
                                       [ymin, ymax], 
                                       [zmin, zmax]])


def distgradient(img):
	fx = np.zeros(shape=img.shape)
	fy = np.zeros(shape=img.shape)
	fz = np.zeros(shape=img.shape)

	J = np.zeros(shape=[ s + 2 for s in img.shape]) # Padded Image
	J[:,:,:] = img.max()
	J[1:-1, 1:-1, 1:-1] = img
	Ne=[[-1, -1, -1], [-1, -1,  0], [-1, -1,  1], [-1,  0, -1], [-1,  0,  0], [-1,  0,  1], [-1,  1, -1], [-1,  1,  0], [-1, 1, 1],       
        [ 0, -1, -1], [ 0, -1,  0], [ 0, -1,  1], [ 0,  0, -1],               [ 0,  0,  1], [ 0,  1, -1], [ 0,  1,  0], [ 0, 1, 1],
        [ 1, -1, -1], [ 1, -1,  0], [ 1, -1,  1], [ 1,  0, -1], [ 1,  0,  0], [ 1,  0,  1], [ 1,  1, -1], [ 1,  1,  0], [ 1, 1, 1]]

	for n in Ne:
		In = J[1+n[0]:J.shape[0]-1+n[0], 1+n[1]:J.shape[1]-1+n[1], 1+n[2]:J.shape[2]-1+n[2]]
		check = In < img;
		img[check] = In[check]
		D = np.divide(n, np.linalg.norm(n))
		fx[check] = D[0]
		fy[check] = D[1]
		fz[check] = D[2]
	return -fx, -fy, -fz


def rivulet_preprocessing(img, config):
    img, cropregion = crop(img, config['threshold']) # Crop it
    bimg = (img > config['threshold']).astype(int)
    bimg = bimg.astype('int')

    # Distance transform from the background
    if not config['silence']: print('Distance Transform...')
    dt = skfmm.distance(bimg, dx=5e-2)
    dt[bimg==0] = 0
    dtmax = dt.max()
    marchmap = np.ones(bimg.shape)
    maxdpt = np.asarray(np.unravel_index(dt.argmax(), dt.shape))
    marchmap[maxdpt[0], maxdpt[1], maxdpt[2]] = -1

    # Fast marching from the position with the largest distance
    if not config['silence']: print('Fast Marching...')
    F = dt ** 4
    F[F == 0] = 1e-10
    t = skfmm.travel_time(marchmap, F, dx=0.01)
    
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

    return dt, t, ginterp, bimg, cropregion
