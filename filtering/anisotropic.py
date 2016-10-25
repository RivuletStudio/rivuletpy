import numpy as np
from scipy.special import jv # Bessel Function of the first kind
from scipy.linalg import eig
from scipy.fftpack import fftn, ifftn, ifft
# import progressbar
from tqdm import tqdm
from scipy.ndimage import filters as fi
import math

# An implementation of the Optimally Oriented 
# M.W.K. Law and A.C.S. Chung, ``Three Dimensional Curvilinear 
# Structure Detection using Optimally Oriented Flux'', ECCV 2008, pp.
# 368--382.
# Max W. K. Law et al., ``Dilated Divergence based Scale-Space 
# Representation for Curve Analysis'', ECCV 2012, pp. 557--571.
# Author: Siqi Liu

def response(img, rsptype='oof', **kwargs):
    eps = 1e-12
    rsp = np.zeros(img.shape)
    # bar = progressbar.ProgressBar(max_value=kwargs['radii'].size)
    # bar.update(0)

    W = np.zeros((img.shape[0], img.shape[1], img.shape[2], 3)) # Eigen values to save
    V = np.zeros((img.shape[0], img.shape[1], img.shape[2], 3, 3)) # Eigen vectors to save

    if rsptype == 'oof' :
        rsptensor = ooftensor(img, kwargs['radii'], kwargs['memory_save'])
    elif rsptype == 'bg':
        rsptensor = bgtensor(img, kwargs['radii'], kwargs['rho'])

    pbar = tqdm(total=len(kwargs['radii']))
    for i, tensorfield in enumerate(rsptensor):
        # Make the tensor from tensorfield
        f11, f12, f13, f22, f23, f33 = tensorfield
        tensor = np.stack((f11, f12, f13, f12, f22, f23, f13, f23, f33), axis=-1)
        del f11
        del f12
        del f13
        del f22
        del f23
        del f33
        tensor = tensor.reshape(img.shape[0], img.shape[1], img.shape[2], 3, 3)
        w, v = np.linalg.eigh(tensor)
        del tensor
        sume = w.sum(axis=-1)
        nvox = img.shape[0] * img.shape[1] * img.shape[2]
        sortidx = np.argsort(np.abs(w), axis=-1)
        sortidx = sortidx.reshape((nvox, 3))

        # Sort eigenvalues according to their abs
        w = w.reshape((nvox, 3))
        for j, (idx, value) in enumerate(zip(sortidx, w)):
            w[j,:] = value[idx]
        w = w.reshape(img.shape[0], img.shape[1], img.shape[2], 3)

        # Sort eigenvectors according to their abs
        v = v.reshape((nvox, 3, 3))
        for j, (idx, vec) in enumerate(zip(sortidx, v)):
            v[j,:,:] = vec[:, idx]
        del sortidx
        v = v.reshape(img.shape[0], img.shape[1], img.shape[2], 3, 3)

        mine = w[:,:,:, 0]
        mide = w[:,:,:, 1]
        maxe = w[:,:,:, 2]

        if rsptype == 'oof':
            feat = maxe
        elif rsptype == 'bg':
            feat = -mide / maxe * (mide + maxe) # Medialness measure response
            cond = sume >= 0
            feat[cond] = 0 # Filter the non-anisotropic voxels

        del mine
        del maxe
        del mide
        del sume

        cond = np.abs(feat) > np.abs(rsp)
        W[cond, :] = w[cond, :]
        V[cond, :, :] = v[cond, :, :]
        rsp[cond] = feat[cond]
        del v
        del w
        del tensorfield
        del feat
        del cond
        pbar.update(1)

    return rsp, V, W


def bgkern3(kerlen, mu=0, sigma=3., rho=0.2):
    '''
    Generate the bi-gaussian kernel
    '''
    sigma_b = rho * sigma
    k = rho ** 2
    kr = (kerlen - 1) / 2 
    X, Y, Z = np.meshgrid(np.arange(-kr, kr+1),
                          np.arange(-kr, kr+1), 
                          np.arange(-kr, kr+1))
    dist = np.linalg.norm(np.stack((X, Y, Z)), axis=0) 

    G  = gkern3(dist, mu, sigma) # Normal Gaussian with mean at origin
    Gb = gkern3(dist, sigma-sigma_b, sigma_b)

    c0 = k * Gb[0, 0, math.floor(sigma_b)] - G[0, 0, math.floor(sigma)]
    c1 = G[0, 0, math.floor(sigma)] - k * Gb[0, 0, math.floor(sigma_b)] + c0
    G += c0
    Gb = k * Gb + c1 # Inverse Gaussian with phase shift

    # Replace the centre of Gb with G
    central_region = dist <= sigma
    del dist
    X = (X[central_region] + kr).astype('int')
    Y = (Y[central_region] + kr).astype('int')
    Z = (Z[central_region] + kr).astype('int')
    Gb[X, Y, Z] = G[X, Y, Z]

    return Gb


def eigh(a, UPLO='L'):
    # I Borrowed from Dipy
    """Iterate over `np.linalg.eigh` if it doesn't support vectorized operation
    Parameters
    ----------
    a : array_like (..., M, M)
        Hermitian/Symmetric matrices whose eigenvalues and
        eigenvectors are to be computed.
    UPLO : {'L', 'U'}, optional
        Specifies whether the calculation is done with the lower triangular
        part of `a` ('L', default) or the upper triangular part ('U').
    Returns
    -------
    w : ndarray (..., M)
        The eigenvalues in ascending order, each repeated according to
        its multiplicity.
    v : ndarray (..., M, M)
        The column ``v[..., :, i]`` is the normalized eigenvector corresponding
        to the eigenvalue ``w[..., i]``.
    Raises
    ------
    LinAlgError
        If the eigenvalue computation does not converge.
    See Also
    --------
    np.linalg.eigh
    """
    a = np.asarray(a)
    if a.ndim > 2 and NUMPY_LESS_1_8:
        shape = a.shape[:-2]
        a = a.reshape(-1, a.shape[-2], a.shape[-1])
        evals = np.empty((a.shape[0], a.shape[1]))
        evecs = np.empty((a.shape[0], a.shape[1], a.shape[1]))
        for i, item in enumerate(a):
            evals[i], evecs[i] = np.linalg.eigh(item, UPLO)
        return (evals.reshape(shape + (a.shape[1], )),
                evecs.reshape(shape + (a.shape[1], a.shape[1])))
    return np.linalg.eigh(a, UPLO)


def gkern3(dist, mu=0., sigma=3.):
    '''
    Make 3D gaussian kernel
    '''
    # Make a dirac spherical function
    return np.exp(-0.5 * (((dist - mu) / sigma)**2)) / (sigma * np.sqrt(2. * np.pi))


def hessian3(x):
    """
    Calculate the hessian matrix with finite differences
    Parameters:
       - x : ndarray
    Returns:
       an array of shape (x.dim, x.ndim) + x.shape
       where the array[i, j, ...] corresponds to the second derivative x_ij
    """
    x_grad = np.gradient(x)
    tmpgrad = np.gradient(x_grad[0])
    f11 = tmpgrad[0]
    f12 = tmpgrad[1]
    f13 = tmpgrad[2]
    tmpgrad = np.gradient(x_grad[1])
    f22 = tmpgrad[1]
    f23 = tmpgrad[2]
    tmpgrad = np.gradient(x_grad[2])
    f33 = tmpgrad[2]
    return [f11, f12, f13, f22, f23, f33]


def bgtensor(img, lsigma, rho=0.2):
    eps = 1e-12
    fimg = fftn(img, overwrite_x=True)

    for s in lsigma:
        jvbuffer = bgkern3(kerlen=math.ceil(s)*6+1, sigma=s, rho=rho)
        jvbuffer = fftn(jvbuffer, shape=fimg.shape, overwrite_x=True) * fimg
        fimg = ifftn(jvbuffer, overwrite_x=True)
        yield hessian3(np.real(fimg))


def eigval33(tensorfield):
    ''' Calculate the eigenvalues of massive 3x3 real symmetric matrices. '''
    a11, a12, a13, a22, a23, a33 = tensorfield  
    eps = 1e-50
    b = a11 + eps
    d = a22 + eps
    j = a33 + eps
    c = - a12**2. - a13**2. - a23**2. + b * d + d * j + j* b 
    d = - b * d * j + a23**2. * b + a12**2. * j - a13**2. * d + 2. * a13 * a12 * a23
    b = - a11 - a22 - a33 - 3. * eps 
    d = d + (2. * b**3. - 9. * b * c) / 27

    c = b**2. / 3. - c
    c = c**3.
    c = c / 27
    c[c < 0] = 0
    c = np.sqrt(c)

    j = c ** (1./3.) 
    c = c + (c==0).astype('float')
    d = -d /2. /c
    d[d>1] = 1
    d[d<-1] = 1
    d = np.real(np.arccos(d) / 3.)
    c = j * np.cos(d)
    d = j * np.sqrt(3.) * np.sin(d)
    b = -b / 3.

    j = -c - d + b
    d = -c + d + b
    b = 2. * c + b

    return b, j, d


def oofftkernel(kernel_radius, r, sigma=1, ntype=1):
    eps = 1e-12
    normalisation = 4/3 * np.pi * r**3 / (jv(1.5, 2*np.pi*r*eps) / eps ** (3/2)) / r**2 *  \
                    (r / np.sqrt(2.*r*sigma - sigma**2)) ** ntype
    jvbuffer = normalisation * np.exp( (-2 * sigma**2 * np.pi**2 * kernel_radius**2) / (kernel_radius**(3/2) ))
    return (np.sin(2 * np.pi * r * kernel_radius) / (2 * np.pi * r * kernel_radius) - np.cos(2 * np.pi * r * kernel_radius)) * \
               jvbuffer * np.sqrt( 1./ (np.pi**2 * r *kernel_radius ))


def ooftensor(img, radii, memory_save=True):
    '''
    type: oof, bg
    '''
    # sigma = 1 # TODO: Pixel spacing
    eps = 1e-12
    # ntype = 1 # The type of normalisation
    fimg = fftn(img, overwrite_x=True)
    shiftmat = ifftshiftedcoormatrix(fimg.shape)
    x, y, z = shiftmat
    x = x / fimg.shape[0]
    y = y / fimg.shape[1]
    z = z / fimg.shape[2]
    kernel_radius = np.sqrt(x ** 2 + y ** 2 + z ** 2) + eps # The distance from origin

    for r in radii:
        # Make the fourier convolutional kernel
        jvbuffer = oofftkernel(kernel_radius, r) * fimg

        if memory_save:
            # F11
            buffer = ifftshiftedcoordinate(img.shape, 0) ** 2 * x * x * jvbuffer
            buffer = ifft(buffer, axis=0)
            buffer = ifft(buffer, axis=1)
            buffer = ifft(buffer, axis=2)
            f11 = buffer.copy()

            # F12
            buffer = ifftshiftedcoordinate(img.shape, 0) * ifftshiftedcoordinate(img.shape, 1) * x * y * jvbuffer
            buffer = ifft(buffer, axis=0)
            buffer = ifft(buffer, axis=1)
            buffer = ifft(buffer, axis=2)
            f12 = buffer.copy()

            # F13
            buffer = ifftshiftedcoordinate(img.shape, 0) * ifftshiftedcoordinate(img.shape, 2) * x * z * jvbuffer
            buffer = ifft(buffer, axis=0)
            buffer = ifft(buffer, axis=1)
            buffer = ifft(buffer, axis=2)
            f13 = buffer.copy()

            # F22
            buffer = ifftshiftedcoordinate(img.shape, 1) ** 2 * y ** 2 * jvbuffer
            buffer = ifft(buffer, axis=0)
            buffer = ifft(buffer, axis=1)
            buffer = ifft(buffer, axis=2)
            f22 = buffer.copy()

            # F23
            buffer = ifftshiftedcoordinate(img.shape, 1) * ifftshiftedcoordinate(img.shape, 2) * y * z * jvbuffer
            buffer = ifft(buffer, axis=0)
            buffer = ifft(buffer, axis=1)
            buffer = ifft(buffer, axis=2)
            f23 = buffer.copy()

            # F33
            buffer = ifftshiftedcoordinate(img.shape, 2) * ifftshiftedcoordinate(img.shape, 2) * z * z * jvbuffer
            buffer = ifft(buffer, axis=0)
            buffer = ifft(buffer, axis=1)
            buffer = ifft(buffer, axis=2)
            f33 = buffer.copy()
        else:
            f11 = np.real(ifftn(x * x * jvbuffer))
            f12 = np.real(ifftn(x * y * jvbuffer))
            f13 = np.real(ifftn(x * z * jvbuffer))
            f22 = np.real(ifftn(y * y * jvbuffer))
            f23 = np.real(ifftn(y * z * jvbuffer))
            f33 = np.real(ifftn(z * z * jvbuffer))
        yield [f11, f12, f13, f22, f23, f33]


# The dimension is a vector specifying the size of the returned coordinate
# matrices. The number of output argument is equals to the dimensionality
# of the vector "dimension". All the dimension is starting from "1"
def ifftshiftedcoormatrix(shape):
    shape = np.asarray(shape)
    p = np.floor(np.asarray(shape) / 2).astype('int')
    coord = []
    for i in range(shape.size):
        a = np.hstack((np.arange(p[i], shape[i]), np.arange(0, p[i]))) - p[i] - 1.
        repmatpara = np.ones((shape.size,)).astype('int')
        repmatpara[i] = shape[i]
        A = a.reshape(repmatpara)
        repmatpara = shape.copy()
        repmatpara[i] = 1
        coord.append(np.tile(A, repmatpara))

    return coord


def ifftshiftedcoordinate(shape, axis):
    shape = np.asarray(shape)
    p = np.floor(np.asarray(shape) / 2).astype('int')
    a = (np.hstack((np.arange(p[axis], shape[axis]), np.arange(0, p[axis]))) - p[axis] - 1.).astype('float')
    a /= shape[axis].astype('float')
    reshapepara = np.ones((shape.size,)).astype('float');
    reshapepara[axis] = shape[axis];
    A = a.reshape(reshapepara);
    repmatpara = shape.copy();
    repmatpara[axis] = 1;
    return np.tile(A, repmatpara)


def nonmaximal_suppression3(img, evl, evt, radius, threshold=0):
    '''
    Non-maximal suppression with oof eigen vector
    img: The input image or filter response
    V: The eigenvector generated by anisotropic filtering algorithm
    radius: the radius to consider for suppression
    '''

    # THE METHOD with ROTATED STENCILS -- Deprecated for now
    # basevec = np.asarray([0., 0., 1.])
    # basestencils = np.asarray([[1., 0., 0.],
    #                 [1./np.sqrt(2.), 1./np.sqrt(2.), 0.],
    #                 [0., 1., 0.],
    #                 [-1./np.sqrt(2.), 1./np.sqrt(2.), 0.],
    #                 [-1, 0., 0.],
    #                 [-1./np.sqrt(2.), -1./np.sqrt(2.), 0.],
    #                 [0., -1, 0.],
    #                 [1./np.sqrt(2.), -1./np.sqrt(2.), 0.]])

    # fg = np.argwhere(img > threshold)
    # nvox = fg.shape[0]

    # imgnew = img.copy()

    # for idx in fg:
    #     ev = V[idx[0], idx[1], idx[2], :, -1]
    #     # Get rotation matrix (http://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d)
    #     v = np.cross(basevec, ev)
    #     s = np.linalg.norm(v)
    #     c = np.dot(basevec, ev)
    #     skw = np.asarray([[0., -v[2], v[1]],
    #                       [v[2], 0, -v[0]],
    #                       [-v[1], v[0], 0.]])
    #     R = np.identity(3) + skw + skw**2 * ((1.-c)/s**2)

    #     # Rotate the stencils
    #     rotstencils = np.dot(R, basestencils.T)

    #     # Get neighbours based on the rotated stencils
    #     neighbours = rotstencils.T + np.tile(idx, (basestencils.shape[0], 1))

    #     # suppress the current image if it is not local maxima
    #     neighbourvox = [img[n[0], n[1], n[2]] for n in neighbours if ] # TODO check in bound
    #     if np.any(neighbourvox > img[idx[0], idx[1], idx[2]]):
    #         imgnew[idx[0], idx[1], idx[2]] = 0.

    suppressed = img.copy()
    suppressed[suppressed <= threshold] = 0
    suppressed_ctr = -1

    while suppressed_ctr is not 0:
        suppressed_ctr = 0
        fgidx = np.argwhere(suppressed > threshold) # Find foreground voxels 

        while fgidx.shape[0] > 0:
            randidx = np.random.randint(0, fgidx.shape[0])
            v = fgidx[randidx, :] # Randomly choose a foreground voxel
            fgidx = np.delete(fgidx, randidx, 0)

            e = evt[v[0], v[1], v[2], :, 0] # The primary eigenvector on v

            # Select the voxels on the orthogonal plane of eigenvector and within a distance to v
            vtile = np.tile(v, (fgidx.shape[0], 1))
            etile = np.tile(e, (fgidx.shape[0], 1))
            cond1 = np.abs(etile * (fgidx - vtile)).sum(axis=-1) < (1.5 * np.sqrt(6) / 4.) # http://math.stackexchange.com/questions/82151/find-the-equation-of-the-plane-passing-through-a-point-and-a-vector-orthogonal
            radius = (evl[v[0], v[1], v[2], :]).sum()
            cond2 = np.linalg.norm(vtile - fgidx, axis=-1) < radius
            cond = np.logical_and(cond1, cond2)
            l = fgidx[cond, :] 

            if l.size == 0:
                continue

            lv = np.asarray([suppressed[l[i, 0], l[i, 1], l[i, 2]] for i in range(l.shape[0])]) # The voxel values of the voxels in l

            if lv.max() > suppressed[v[0], v[1], v[2]]:
                suppressed[v[0], v[1], v[2]] = 0
                suppressed_ctr += 1 

    return suppressed 