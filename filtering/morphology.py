import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage.filters import laplace
try:
    from skimage import filters
except ImportError:
    from skimage import filter as filters
from tqdm import tqdm
from functools import reduce
from scipy.interpolate import RegularGridInterpolator
import skfmm


def ssmdt(dt, ssmiter):
    dt = ssm(dt, anisotropic=True, iterations=ssmiter)
    dt[dt < filters.threshold_otsu(dt)] = 0
    dt = skfmm.distance(dt, dx=5e-2)
    dt = skfmm.distance(np.logical_not(dt), dx=5e-3)
    dt[dt > 0.04] = 0.04
    dt = dt.max() - dt
    dt[dt <= 0.038] = 0
    return dt


def ssm(img, anisotropic=False, iterations=30):
    '''
    Skeleton strength map
    img: the input image
    anisotropic: True if using anisotropic diffusion
    iterations: number of iterations to optimise the GVF
    '''
    # f = gaussian_gradient_magnitude(img, 1)
    # f = 1 - gimg # Inverted version of the smoothed
    # gradient of the distance transform

    gvfmap = gvf(img, mu=0.001, iterations=iterations, anisotropic=anisotropic)

    shifts = [-1, 1, -1, -1, -1, 1]
    axis = [1, 1, 2, 2, 3, 3]
    shiftmat = np.zeros(gvfmap.shape)
    f = np.zeros(img.shape)  # reuse f for saving the SSM

    for i, (s, a) in enumerate(zip(shifts, axis)):
        # Only the orthogonal neighbours
        shiftmat.fill(0)
        shiftmat[a - 1, :, :, :] = s

        # Dot product gvf and neighbour displacement fields /
        # distance between neighbour
        f += np.sum(np.roll(
            gvfmap, s, axis=a) * shiftmat, axis=0) / np.linalg.norm(
                shiftmat, axis=0)

    f[np.isnan(f)] = 0
    f[f < 0] = 0
    return f


def nonmax(img, sigma=2, threshold=0):
    '''
    Finds directional local maxima
    in a gradient array, as used in the Canny edge detector, but made
    separately accessible here for greater flexibility. The result is a
    logical array with the value true where the gradient magnitude is a
    local maximum along the gradient direction.
    '''

    # Get normalised gaussian gradients
    eps = 1e-12
    gx = gaussian_filter1d(img, sigma, axis=0, order=1)
    gy = gaussian_filter1d(img, sigma, axis=1, order=1)
    gz = gaussian_filter1d(img, sigma, axis=2, order=1)
    gmag = np.sqrt(gx**2 + gy**2 + gz**2)

    gx = gx / (gmag + eps)
    gy = gy / (gmag + eps)
    gz = gz / (gmag + eps)
    standard_grid = (np.arange(gmag.shape[0]), np.arange(gmag.shape[1]),
                     np.arange(gmag.shape[2]))
    ginterp = RegularGridInterpolator(standard_grid, gmag, bounds_error=False)

    # Interpolate the graident magnitudes
    idx = np.argwhere(
        img > threshold
    )  # Double-check if the original image should be used to check
    xidx = idx[:, 0]
    yidx = idx[:, 1]
    zidx = idx[:, 2]
    dx = gx[xidx, yidx, zidx]
    dy = gy[xidx, yidx, zidx]
    dz = gz[xidx, yidx, zidx]
    gmag_0 = gmag[xidx, yidx, zidx]
    gmag_1 = ginterp(np.stack((xidx + dx, yidx + dy, zidx + dz), axis=-1))
    gmag_2 = ginterp(np.stack((xidx - dx, yidx - dy, zidx - dz), axis=-1))

    # Suppress nonmax voxels
    keep = np.logical_and(gmag_0 > gmag_1, gmag_0 > gmag_2)
    gmag.fill(0)
    gmag[xidx, yidx, zidx] = keep.astype('float')

    return gmag


def d(x):
    '''
     The difference between a voxel and its six neighbours
     The boundary is messed, however it does not matter
    '''
    diff = np.zeros((6, x.shape[0], x.shape[1], x.shape[2]))
    shifts = [-1, 1, -1, -1, -1, 1]
    axis = [0, 0, 1, 1, 2, 2]

    for i, (s, a) in enumerate(zip(shifts, axis)):
        diff[i] = np.roll(x, s, axis=a) - x

    return diff


# The decreasing funciton for angles
def g_all(u, v, w):
    G = np.zeros((6, u.shape[0], u.shape[1], u.shape[2]))  # Result
    cvec = np.stack((u, v, w), axis=0)  # The flow vector on central voxel
    shifts = [-1, 1, -1, -1, -1, 1]
    axis = [0, 0, 1, 1, 2, 2]

    for i, (s, a) in enumerate(zip(shifts, axis)):
        G[i] = g(
            cvec,
            np.stack(
                (
                    np.roll(
                        u, s,
                        axis=a),  # The flow vector on the surronding voxel
                    np.roll(
                        v, s, axis=a),
                    np.roll(
                        w, s, axis=a)),
                axis=0))
    return G


# Calculate the G function between two vector fields
def g(cvec, svec, K=1):
    cnorm = np.linalg.norm(cvec, axis=0)
    snorm = np.linalg.norm(svec, axis=0)
    t = np.sum(cvec * svec, axis=0) / (cnorm * snorm + 1e-12)
    t -= 1
    t = np.exp(K * t)
    t[np.logical_or(cnorm == 0, snorm == 0)] = 0

    return t


# Divergence
def div(x):
    """ compute the divergence of n-D scalar field `F` """
    return reduce(
        np.add, np.gradient(x))  # http://stackoverflow.com/a/21134289/1890513


def gvf(f, mu=0.05, iterations=30, anisotropic=False,
        ignore_second_term=False):
    # Gradient vector flow
    # Translated from https://github.com/smistad/3D-Gradient-Vector-Flow-for-Matlab
    f = (f - f.min()) / (f.max() - f.min())
    f = enforce_mirror_boundary(
        f)  # Enforce the mirror conditions on the boundary

    dx, dy, dz = np.gradient(f)  # Initialse with normal gradients
    '''
    Initialise the GVF vectors following S3 in
    Yu, Zeyun, and Chandrajit Bajaj. 
    "A segmentation-free approach for skeletonization of gray-scale images via anisotropic vector diffusion." 
    CVPR, 2004. CVPR 2004. 
    It only uses one of the surronding neighbours with the lowest intensity
    '''
    magsq = dx**2 + dy**2 + dz**2

    # Set up the initial vector field
    u = dx.copy()
    v = dy.copy()
    w = dz.copy()

    for i in tqdm(range(iterations)):
        # The boundary might not matter here
        # u = enforce_mirror_boundary(u)
        # v = enforce_mirror_boundary(v)
        # w = enforce_mirror_boundary(w)

        # Update the vector field
        if anisotropic:
            G = g_all(u, v, w)
            u += mu / 6. * div(np.sum(G * d(u), axis=0))
            v += mu / 6. * div(np.sum(G * d(v), axis=0))
            w += mu / 6. * div(np.sum(G * d(w), axis=0))
        else:
            u += mu * 6 * laplace(u)
            v += mu * 6 * laplace(v)
            w += mu * 6 * laplace(w)

        if not ignore_second_term:
            u -= (u - dx) * magsq
            v -= (v - dy) * magsq
            w -= (w - dz) * magsq

    return np.stack((u, v, w), axis=0)


def enforce_mirror_boundary(f):
    '''
    This function enforces the mirror boundary conditions
    on the 3D input image f. The values of all voxels at
    the boundary is set to the values of the voxels 2 steps
    inward
    '''
    N, M, O = f.shape

    # Indices in the middle
    xi = np.arange(1, M - 1)
    yi = np.arange(1, N - 1)
    zi = np.arange(1, O - 1)

    # Corners
    f[[0, -1], [0, -1], [0, -1]] = f[[2, -3], [2, -3], [2, -3]]

    # Edges
    f[np.ix_([0, -1], [0, -1], zi)] = f[np.ix_([2, -3], [2, -3], zi)]
    f[np.ix_(yi, [0, -1], [0, -1])] = f[np.ix_(yi, [2, -3], [2, -3])]
    f[np.ix_([0, -1], xi, [0, -1])] = f[np.ix_([2, -3], xi, [2, -3])]

    # Faces
    f[np.ix_([0, -1], xi, zi)] = f[np.ix_([2, -3], xi, zi)]
    f[np.ix_(yi, [0, -1], zi)] = f[np.ix_(yi, [2, -3], zi)]
    f[np.ix_(yi, xi, [0, -1])] = f[np.ix_(yi, xi, [2, -3])]
    return f
