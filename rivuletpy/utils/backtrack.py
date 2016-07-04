import numpy as np
import random
import math
from euclid import Point3

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
	                max(z-r, 0) : min(z+r+1, bimg.shape[2])].sum() / (2*r + 1)**3 < 0.5:
				break
		except IndexError:
			break

	return r


def inbound(pt, shape):
	return all([True if 0 <= p < s else False for p,s in zip(pt, shape)])


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

        points.append(Point3(x, y, z))

    return points
