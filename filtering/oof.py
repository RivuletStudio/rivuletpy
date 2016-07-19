import numpy as np
from scipy.special import jv # Bessel Function of the first kind
from scipy.linalg import eig
from scipy.fftpack import fftn, ifftn, ifft
import progressbar

# An implementation of the Optimally Oriented 
# M.W.K. Law and A.C.S. Chung, ``Three Dimensional Curvilinear 
# Structure Detection using Optimally Oriented Flux'', ECCV 2008, pp.
# 368--382.
# Max W. K. Law et al., ``Dilated Divergence based Scale-Space 
# Representation for Curve Analysis'', ECCV 2012, pp. 557--571.
# Author: Siqi Liu

def oofresponse(img, radii, memory_save=True):
    rsp = np.zeros(img.shape)
    bar = progressbar.ProgressBar(max_value=radii.size)

    for i,tensorfield in enumerate(anisotropic_tensor(img, radii, oofftkernel, memory_save)):
        eig1, eig2, eig3 = eigval33(tensorfield)
        maxe = eig1
        mine = eig1
        mide = maxe + eig2 + eig3   

        maxe[np.abs(eig2) > np.abs(maxe)] = eig2[ np.abs(eig2) > np.abs(maxe) ]
        mine[np.abs(eig2) < np.abs(mine)] = eig2[ np.abs(eig2) < np.abs(mine) ]
        maxe[np.abs(eig3) > np.abs(maxe)] = eig3[ np.abs(eig3) > np.abs(maxe) ]
        mine[np.abs(eig3) < np.abs(mine)] = eig3[ np.abs(eig3) < np.abs(mine) ]
        
        feat = maxe
        cond = np.abs(feat) > np.abs(rsp)
        rsp[cond] = feat[cond]
        bar.update(i+1)

    return rsp


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


def anisotropic_tensor(img, radii, kernelfunc, memory_save=True):
    # sigma = 1 # TODO: Pixel spacing
    eps = 1e-12
    # ntype = 1 # The type of normalisation
    fimg = fftn(img)
    shiftmat = ifftshiftedcoormatrix(fimg.shape)
    x, y, z = shiftmat
    x = x / fimg.shape[0]
    y = y / fimg.shape[1]
    z = z / fimg.shape[2]
    kernel_radius = np.sqrt(x ** 2 + y ** 2 + z ** 2) + eps # Should be the radius of the kernel
    tensor = np.zeros((img.shape[0], img.shape[1], img.shape[2], 6))

    for r in radii:
        # Make the fourier convolutional kernel
        jvbuffer = kernelfunc(kernel_radius, r) * fimg # Clear radius

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
