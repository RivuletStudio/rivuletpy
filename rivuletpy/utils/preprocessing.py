import numpy as np
from .io import *
from scipy.interpolate import RegularGridInterpolator

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


