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

    return radius > mindist, minidx


def rotate(hsp, axis, angle):
    '''
    Rotate a 3D mesh with an angle according to an arbitrary axis
    Implemented following:
    http://www.siggraph.org/education/materials/HyperGraph/modeling/mod_tran/3drota.htm
    '''

    hsp = np.vstack((hsp[0], hsp[1], hsp[2], [1.]*hsp[0].size))
    a, b, c = axis
    d = 1.
    rxa = np.asarray([[0., 0., 0., 0.],
                      [0., c/d, b/a, 0.], 
                      [0.,-b/a,c/a,0.], 
                      [0.,0.,0.,1.]])
        
    ryb = np.asarray([[d, 0., a, 0.],
                      [0., 1., 0., 0.],
                      [-a, 0., d, 0.],
                      [0., 0., 0., 1.]])

    rzq = np.asarray([[np.cos(angle), np.sin(angle), 0., 0.],
                      [-np.sin(angle), np.cos(angle), 0., 0.],
                      [0., 0., 1., 0.],
                      [0., 0., 0., 1.]])

    R = rxa * ryb * rzq * ryb.T * rxa.T
    hsp = R.dot(hsp)
    print('==hsp shape:', hsp.shape)

    return hsp[0, :].flatten(), hsp[1, :].flatten(), hsp[2, :].flatten()


def genrotmesh(vec=np.asarray([0., 0., 1.]), pos=np.asarray([0., 0., 0.]), radius=30.):
    eps = 1e-12
    ngrid = 2 * radius + 1
    hspx, hspy = np.meshgrid(np.linspace(0. - radius, 0. + radius, ngrid),
                      np.linspace(0. - radius, 0. + radius, ngrid))
    hspz = np.asarray([0.] * hspx.size)
    hsp = [hspx.flatten(), hspy.flatten(), hspz]

    hspvec = np.asarray([0., 0., 1.]) 
    hspvec = hspvec / np.linalg.norm(hspvec)
    hspvec[hspvec==0] = eps

    vec = np.asarray(vec) 
    vec = vec / np.linalg.norm(vec)
    vec[vec==0] = eps

    # Find how much rotation is needed, below are the references.
    # http://www.siggraph.org/education/materials/HyperGraph/modeling/mod_tran/3drota.htm
    # http://www.mathworks.com/help/toolbox/sl3d/vrrotvec.html
    hspvecx = np.cross(hspvec, vec) 
    hspvecx /= np.linalg.norm(hspvecx)
    acosineval = np.arccos(hspvec * vec)
    hspx, hspy, hspz = rotate(hsp, hspvecx, acosineval) # Rotate an angle according to axis hspvecx
    hspx += pos[0]
    hspy += pos[1]
    hspz += pos[2]
    print('== hspx shape:', hspx.shape)
    print('== hspy shape:', hspy.shape)
    print('== hspz shape:', hspz.shape)
    
    return hspx, hspy, hspz


def extract_slice(I, vec=np.asarray([0., 0., 1.]), pos=np.asarray([0., 0., 0.]), radius=10):
    # ExtractSlice extracts an arbitray slice from a volume.
    # Adapted from: http://www.mathworks.com/matlabcentral/fileexchange/32032-extract-slice-from-volume
    hsp = genrotmesh(vec, pos, radius)
    standard_grid = (np.arange(I.shape[0]), np.arange(I.shape[1]), np.arange(I.shape[2]))
    imginterp = RegularGridInterpolator(standard_grid, I, method='nearest')
    imslice = imginterp(hsp[0], hsp[1], hsp[2])
    return imslice.reshape(ngrid, ngrid)
