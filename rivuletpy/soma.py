# -*- coding: utf-8 -*-
"""
somasnakes
===========
Original package is adjusted for soma detection by donghaozhang and siqiliu.

This soma submodule can be used for soma detection only, but this submodule is
currently embedded in rivuletpy. The soma mask can be generate by setting
its corresponding argument. Soma detection requires an initial soma centroid,
estimated somatic radius and grayscale neuron image. Soma growth is based on
the Morphological Active Contours without Edges algorithm.  The original paper
is named as A morphological approach to curvature-based
evolution of curves and surfaces.The following papers are Rivulet papers.
The soma growth algorithm can converge by applying the sliding window.
Journal Rivulet Paper : Rivulet: 3D Neuron Morphology Tracing
with Iterative Back-Tracking Conference Rivulet Paper : Reconstruction
 of 3D neuron morphology using Rivulet back-tracking
soma is a submodule of rivuletpy
"""

__author__ = "Donghao Zhang <zdhpeter1991@gmail.com>, Siqi Liu <lsqshr@gmail.com>"

from itertools import cycle
import math
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion
from scipy.ndimage import gaussian_filter, gaussian_gradient_magnitude
from scipy.ndimage.measurements import center_of_mass
from scipy.ndimage.morphology import generate_binary_structure
import skfmm


class Soma(object):
    def __init__(self):
        self.centroid = None
        self.radius = 0 
        self.mask = None

    def simple_mask(self, bimg):
        '''
        Make soma binary mask with the original
        binary image and its radius and position
        '''

        # Make a ball like mask with 2 X somaradius
        ballvolume = np.zeros(bimg.shape)
        ballvolume[self.centroid[0], self.centroid[1], self.centroid[2]] = 1
        stt = generate_binary_structure(3, 1)
        for i in range(math.ceil(self.radius * 2.5)):
            ballvolume = binary_dilation(ballvolume, structure=stt)

        # Make the soma mask with the intersection
        #between the ball area and the original binary
        self.mask = np.logical_and(ballvolume, bimg)

    # Shift the centroid according to the cropped region
    def crop_centroid(self, crop_region):
        self.centroid[0] = self.centroid[0] - crop_region[0, 0]
        self.centroid[1] = self.centroid[1] - crop_region[1, 0]
        self.centroid[2] = self.centroid[2] - crop_region[2, 0]

    def detect(self, bimg, simple=False, silent=False):
        """
        Automatic detection of soma volume unless the iterations are given.
        """

        # Smooth iterations
        smoothing = 1
        # A float number controls the weight of internal energy
        lambda1 = 1
        # A float number controls the weight of external energy
        lambda2 = 1.5
        # Manually set the number of iterations required for the soma
        # The type of iterations is int
        iterations = -1
        bimg = bimg.astype('int')  # Segment
        dt = skfmm.distance(bimg, dx=1.1)  # Boundary DT

        # somaradius : the approximate value of
        # soma radius estimated from distance transform
        # the type of somaradius is float64
        # somaradius is just a float number
        somaradius = dt.max()

        # somapos : the coordinate of estimated soma centroid
        # the type of somapos is int64
        # the shape of somapos is (3,)
        # somapos is array-like
        somapos = np.asarray(np.unravel_index(dt.argmax(), dt.shape))

        # Soma detection is required
        if not simple:
            if not silent: print('Reconstructing Soma with SRET')
            ratioxz = bimg.shape[0] / bimg.shape[2]
            ratioyz = bimg.shape[1] / bimg.shape[2]
            sqrval = (somaradius**0.5 * max(ratioxz, ratioyz))
            sqrval = np.floor(min(max(sqrval, 3), (somaradius**0.5) * 6))

            startpt = somapos - 3 * sqrval
            endpt = somapos + 3 * sqrval

            # # To constrain the soma growth region inside the cubic region
            # # Python index start from 0
            startpt[0] = min(max(0, startpt[0]), bimg.shape[0] - 1)
            startpt[1] = min(max(0, startpt[1]), bimg.shape[1] - 1)
            startpt[2] = min(max(0, startpt[2]), bimg.shape[2] - 1)

            endpt[0] = min(max(0, endpt[0]), bimg.shape[0] - 1)
            endpt[1] = min(max(0, endpt[1]), bimg.shape[1] - 1)
            endpt[2] = min(max(0, endpt[2]), bimg.shape[2] - 1)
            startpt = startpt.astype(int)  # Convert type to int for indexing
            endpt = endpt.astype(int)

            # # Extract soma region for fast soma detection
            somaimg = bimg[startpt[0]:endpt[0], startpt[1]:endpt[1], startpt[2]:
                           endpt[2]]
            centerpt = np.zeros(3)
            centerpt[0] = somaimg.shape[0] / 2
            centerpt[1] = somaimg.shape[1] / 2
            centerpt[2] = somaimg.shape[2] / 2
            centerpt = np.floor(centerpt)

            # Morphological ACWE. Initialization of the level-set.
            macwe = MorphACWE(somaimg, startpt, endpt, smoothing, lambda1, lambda2)
            macwe.levelset = circle_levelset(somaimg.shape,
                                             np.floor(centerpt), sqrval)

            # -1 means the automatic detection
            # Positive integers means the number of iterations
            if iterations == -1:
                macwe.autoconvg()  # automatic soma detection
            else:
                # Input the iteration number manually
                for i in range(iterations):
                    macwe.step()

            # The following achieves the automatic somtic box extension
            # The maximum somatic region extension iteration
            # It is set to 10 avoid infinite loops
            for i in range(1, 11):
                # if not silent:
                    # print('The somatic region extension iteration is', i)
                if macwe.enlrspt is None:
                    break

                # Copy the values to new variables for the safe purpose
                startpt = macwe.enlrspt.copy()
                endpt = macwe.enlrept.copy()
                startpt[0] = min(max(0, startpt[0]), bimg.shape[0])
                startpt[1] = min(max(0, startpt[1]), bimg.shape[1])
                startpt[2] = min(max(0, startpt[2]), bimg.shape[2])

                endpt[0] = min(max(0, endpt[0]), bimg.shape[0])
                endpt[1] = min(max(0, endpt[1]), bimg.shape[1])
                endpt[2] = min(max(0, endpt[2]), bimg.shape[2])
                somaimg = bimg[startpt[0]:endpt[0], startpt[1]:endpt[1], startpt[2]:
                              endpt[2]]
                full_soma_mask = np.zeros((bimg.shape[0], bimg.shape[1], bimg.shape[2]))

                # Put the detected somas into the whole image
                # It is either true or false
                full_soma_mask[macwe.startpoint[0]:macwe.endpoint[
                    0], macwe.startpoint[1]:macwe.endpoint[1], macwe.startpoint[2]:
                            macwe.endpoint[2]] = macwe._u

                # The newlevelset is the initial soma volume from previous iteration
                #(the automatic converge operation)
                newlevelset = full_soma_mask[startpt[0]:endpt[0], startpt[1]:endpt[1],
                                          startpt[2]:endpt[2]]

                # The previous macwe class is released
                # To avoid the conflicts with the new initialisation of the macwe class
                del macwe

                # Initialisation for the new class
                macwe = MorphACWE(somaimg, startpt, endpt, smoothing, lambda1,
                                  lambda2)
                del somaimg, full_soma_mask, startpt, endpt

                # Reuse the soma volume from previous iteration
                macwe.set_levelset(newlevelset)

                # Release memory to avoid conflicts with previous newlevelset
                del newlevelset
                macwe.autoconvg()

            # The automatic smoothing operation to remove the interferes with dendrites
            macwe.autosmooth()

            # Initialise soma mask image
            full_soma_mask = np.zeros((bimg.shape[0], bimg.shape[1], bimg.shape[2]))

            # There are two possible scenarios
            # The first scenrio is that the automatic box extension is not necessary
            if macwe.enlrspt is None:
                startpt = macwe.startpoint.copy()
                endpt = macwe.endpoint.copy()
            # The second scenrio is that the automatic box extension operations has been performed
            else:
                startpt = macwe.enlrspt.copy()
                endpt = macwe.enlrept.copy()

            startpt[0] = min(max(0, startpt[0]), bimg.shape[0])
            startpt[1] = min(max(0, startpt[1]), bimg.shape[1])
            startpt[2] = min(max(0, startpt[2]), bimg.shape[2])

            endpt[0] = min(max(0, endpt[0]), bimg.shape[0])
            endpt[1] = min(max(0, endpt[1]), bimg.shape[1])
            endpt[2] = min(max(0, endpt[2]), bimg.shape[2])
            # The soma mask image contains only two possible values
            # Each element is either 0 or 40
            # Value 40 is assigned for the visualisation purpose.
            full_soma_mask[startpt[0]:endpt[0], startpt[1]:endpt[1], startpt[2]:endpt[
                2]] = macwe._u > 0

            # Calculate the new centroid using the soma volume
            newsomapos = center_of_mass(full_soma_mask)

            # Round the float coordinates into integers
            newsomapos = [math.floor(p) for p in newsomapos]
            self.centroid = newsomapos
            self.radius = somaradius
            self.mask = full_soma_mask
        else:
            if not silent: print('Reconstructing Soma with Simple Mask')
            self.centroid = somapos
            self.radius = somaradius
            self.simple_mask(bimg)


class Fcycle(object):
    def __init__(self, iterable):
        """Call functions from the iterable each time it is called."""
        self.funcs = cycle(iterable)

    def __call__(self, *args, **kwargs):
        f = next(self.funcs)
        return f(*args, **kwargs)


# SI and IS operators for 2D and 3D.
_P2 = [
    np.eye(3), np.array([[0, 1, 0]] * 3), np.flipud(np.eye(3)),
    np.rot90([[0, 1, 0]] * 3)
]
_P3 = [np.zeros((3, 3, 3)) for i in range(9)]

_P3[0][:, :, 1] = 1
_P3[1][:, 1, :] = 1
_P3[2][1, :, :] = 1
_P3[3][:, [0, 1, 2], [0, 1, 2]] = 1
_P3[4][:, [0, 1, 2], [2, 1, 0]] = 1
_P3[5][[0, 1, 2], :, [0, 1, 2]] = 1
_P3[6][[0, 1, 2], :, [2, 1, 0]] = 1
_P3[7][[0, 1, 2], [0, 1, 2], :] = 1
_P3[8][[0, 1, 2], [2, 1, 0], :] = 1

_aux = np.zeros((0))


def SI(u):
    """SI operator."""
    # print('SI operator has been called')
    global _aux
    if np.ndim(u) == 2:
        P = _P2
    elif np.ndim(u) == 3:
        P = _P3
    else:
        raise ValueError(
            "u has an invalid number of dimensions (should be 2 or 3)")

    if u.shape != _aux.shape[1:]:
        _aux = np.zeros((len(P), ) + u.shape)

    for i in range(len(P)):
        _aux[i] = binary_erosion(u, P[i])

    return _aux.max(0)


def circle_levelset(shape, center, sqradius, scalerow=1.0):
    """Build a binary function with a circle as the 0.5-levelset."""
    grid = np.mgrid[list(map(slice, shape))].T - center
    phi = sqradius - np.sqrt(np.sum((grid.T)**2, 0))
    u = np.float_(phi > 0)
    return u


def IS(u):
    """IS operator."""
    global _aux
    if np.ndim(u) == 2:
        P = _P2
    elif np.ndim(u) == 3:
        P = _P3
    else:
        raise ValueError(
            "u has an invalid number of dimensions (should be 2 or 3)")

    if u.shape != _aux.shape[1:]:
        _aux = np.zeros((len(P), ) + u.shape)

    for i in range(len(P)):
        _aux[i] = binary_dilation(u, P[i])

    return _aux.min(0)


# SIoIS operator.
SIoIS = lambda u: SI(IS(u))
ISoSI = lambda u: IS(SI(u))
curvop = Fcycle([SIoIS, ISoSI])

# Stopping factors (function g(I) in the paper).


def gborders(img, alpha=1.0, sigma=1.0):
    """Stopping criterion for image borders."""

    # The norm of the gradient.
    gradnorm = gaussian_gradient_magnitude(img, sigma, mode='constant')
    return 1.0 / np.sqrt(1.0 + alpha * gradnorm)


def glines(img, sigma=1.0):
    """Stopping criterion for image black lines."""
    return gaussian_filter(img, sigma)


class MorphACWE(object):
    """Morphological ACWE based on the Chan-Vese energy functional."""

    def __init__(self,
                 data,
                 startpoint,
                 endpoint,
                 imgshape,
                 smoothing=1,
                 lambda1=1,
                 lambda2=1.5):
        """Create a Morphological ACWE solver.

        Parameters
        ----------
        data : ndarray
            The image data.
        smoothing : scalar
            The number of repetitions of the smoothing step (the
            curv operator) in each iteration. In other terms,
            this is the strength of the smoothing. This is the
            parameter µ.
        lambda1, lambda2 : scalars
            Relative importance of the inside pixels (lambda1)
            against the outside pixels (lambda2).
        startpt, endpt : numpy int array
            startpt is the initial starting point of the somatic region
            endpt is the initial ending point of the somatic region
        """
        self._u = None
        self.smoothing = smoothing
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.imgshape = imgshape
        self.data = data
        self.startpoint = startpoint
        self.endpoint = endpoint
        self.enlrspt = None
        self.enlrept = None

    def set_levelset(self, u):
        self._u = np.double(u)
        self._u[u > 0] = 1
        self._u[u <= 0] = 0

    levelset = property(
        lambda self: self._u,
        set_levelset,
        doc="The level set embedding function (u).")

    def step(self):
        """Perform a single step of the morphological Chan-Vese evolution."""
        # Assign attributes to local variables for convenience.
        u = self._u

        if u is None:
            raise ValueError(
                "the levelset function is not set (use set_levelset)")

        data = self.data

        # Determine c0 and c1.
        inside = u > 0
        outside = u <= 0
        c0 = data[outside].sum() / float(outside.sum())
        c1 = data[inside].sum() / float(inside.sum())

        # Image attachment.
        dres = np.array(np.gradient(u))
        abs_dres = np.abs(dres).sum(0)
        #aux = abs_dres * (c0 - c1) * (c0 + c1 - 2*data)
        aux = abs_dres * (self.lambda1 * (data - c1)**2 - self.lambda2 *
                          (data - c0)**2)

        res = np.copy(u)
        res[aux < 0] = 1
        res[aux > 0] = 0

        res = IS(res)
        # Smoothing.
        for i in range(self.smoothing):
            res = curvop(res)
        self._u = res

    def step_sm(self):
        """A smoothing step of the morphological Chan-Vese evolution."""
        # Assign attributes to local variables for convenience.
        u = self._u

        if u is None:
            raise ValueError(
                "the levelset function is not set (use set_levelset)")
        res = np.copy(u)

        # Smoothing.
        res = curvop(res)
        self._u = res

    def run(self, iterations):
        """Run several iterations of the morphological Chan-Vese method."""
        for i in range(iterations):
            self.step()

    def autoconvg(self):
        """Soma detection converges by itself."""

        # Autoconvg is the abbreviation of automatic convergence
        iterations = 200

        # The following vector is the number of foreground voxels
        foreground_num = np.zeros(iterations)

        # The following vector is initialised for storing forward difference
        forward_diff_store = np.zeros(iterations)

        # This is the initilization of automatic converge
        for i in range(iterations):
            self.step()
            u = self._u
            volu = sum(u[u > 0])
            foreground_num[i] = volu
            if i > 0:
                # The variable diff_step is the current first order difference
                diff_step = foreground_num[i] - foreground_num[i - 1]
                forward_diff_store[i - 1] = diff_step
                if i > 6:
                    # The variable cur_slider_diff is the sum of sliding window
                    # The size of sliding window is 6
                    cur_slider_diff = np.sum(forward_diff_store[i - 6:i - 1])
                    volu_thres = 0.05 * foreground_num[i]
                    convg_one = np.absolute(cur_slider_diff) < 20
                    convg_two = np.absolute(cur_slider_diff) < volu_thres
                    convg_criteria = np.logical_or(convg_one, convg_two)
                    if convg_criteria:
                        break

        A = self._u > 0.5
        slicevalarray = np.zeros(6)

        # Front face along dimension 1
        somaslice = A[0, :, :]
        slicearray = np.sum(somaslice, axis=0)
        sliceval = np.sum(slicearray, axis=0)
        slicevalarray[0] = sliceval

        # Back face along dimension 1
        somaslice = A[A.shape[0] - 1, :, :]
        slicearray = np.sum(somaslice, axis=0)
        sliceval = np.sum(slicearray, axis=0)
        slicevalarray[1] = sliceval

        # Front face along dimension 2
        somaslice = A[:, 0, :]
        slicearray = np.sum(somaslice, axis=0)
        sliceval = np.sum(slicearray, axis=0)
        slicevalarray[2] = sliceval

        # Back face along dimension 2
        somaslice = A[:, A.shape[1] - 1, :]
        slicearray = np.sum(somaslice, axis=0)
        sliceval = np.sum(slicearray, axis=0)
        slicevalarray[3] = sliceval

        # Front face along dimension 3
        somaslice = A[:, :, 0]
        slicearray = np.sum(somaslice, axis=0)
        sliceval = np.sum(slicearray, axis=0)
        slicevalarray[4] = sliceval

        # Back face along dimension 3
        somaslice = A[:, :, A.shape[2] - 1]
        slicearray = np.sum(somaslice, axis=0)
        sliceval = np.sum(slicearray, axis=0)
        slicevalarray[5] = sliceval

        # The maxval is used to compare the threshold(100 mentioned later)
        maxval = slicevalarray.max()

        # The maxind is the index of slicevalarray.
        # In addition, it determines which wall will be extended
        maxind = slicevalarray.argmax()

        # The size of binary data image
        sz1 = self.data.shape[0]
        # sz2 = self.data.shape[1]
        # sz3 = self.data.shape[2]

        # extend = enlrspt have value, not extend = (enlrspt=None)
        # 100 : A threshold of the total number of somatic voxels on each wall
        if (maxval > 100):
            self.enlrspt = self.startpoint.copy()
            self.enlrept = self.endpoint.copy()
            # The following code determines the most possible wall(face)
            # which requires the extension
            if (maxind == 0):
                self.enlrspt[0] = self.enlrspt[0] - (sz1 / 4)
            elif (maxind == 1):
                self.enlrept[0] = self.enlrept[0] + (sz1 / 4)
            elif (maxind == 2):
                self.enlrspt[1] = self.enlrspt[1] - (sz1 / 4)
            elif (maxind == 3):
                self.enlrept[1] = self.enlrept[1] + (sz1 / 4)
            elif (maxind == 4):
                self.enlrspt[2] = self.enlrspt[2] - (sz1 / 4)
            elif (maxind == 5):
                self.enlrept[2] = self.enlrept[2] + (sz1 / 4)

            # To constrain new bounding box inside the image size
        else:
            self.enlrspt = None
            self.enlrept = None

    def autosmooth(self):
        """The automatic smoothing of soma volume to remove dendrites"""

        # The autosmooth is the abbreviation of automatic smoothing
        iterations = 20

        # Calculate the initial volume
        u = self._u
        ini_vol = sum(u[u > 0])

        # The smooth operation make
        for i in range(iterations):
            self.step_sm()
            u = self._u
            volu = sum(u[u > 0])
            vol_pct = volu / ini_vol


            # The criteria of the termination of soma growth
            # The somatic volume underwent dramatic change
            judge_one = vol_pct < 0.75
            judge_two = vol_pct > 1.15
            judge_criteria = np.logical_or(judge_one, judge_two)
            if judge_criteria:
                break


def evolve_visual(msnake, levelset=None, num_iters=20, background=None):
    """
    Visual evolution of a morphological snake.

    Parameters
    ----------
    msnake : MorphGAC or MorphACWE instance
        The morphological snake solver.
    levelset : array-like, optional
        If given, the levelset of the solver is initialized to this. If not
        given, the evolution will use the levelset already set in msnake.
    num_iters : int, optional
        The number of iterations.
    background : array-like, optional
        If given, background will be shown behind the contours instead of
        msnake.data.
    """
    from matplotlib import pyplot as ppl

    if levelset is not None:
        msnake.levelset = levelset

    # Prepare the visual environment.
    fig = ppl.gcf()
    fig.clf()
    ax1 = fig.add_subplot(1, 2, 1)
    if background is None:
        ax1.imshow(msnake.data, cmap=ppl.cm.gray)
    else:
        ax1.imshow(background, cmap=ppl.cm.gray)
    ax1.contour(msnake.levelset, [0.5], colors='r')
    ax2 = fig.add_subplot(1, 2, 2)
    ax_u = ax2.imshow(msnake.levelset)
    ppl.pause(0.001)

    # Iterate.
    for i in range(num_iters):

        # Evolve.
        msnake.step()

        # Update figure.
        del ax1.collections[0]
        ax1.contour(msnake.levelset, [0.5], colors='r')
        ax_u.set_data(msnake.levelset)
        fig.canvas.draw()
        #ppl.pause(0.001)

    # Return the last levelset.
    return msnake.levelset


def evolve_visual3d(msnake, levelset=None, num_iters=20):
    """
    Visual evolution of a three-dimensional morphological snake.

    Parameters
    ----------
    msnake : MorphGAC or MorphACWE instance
        The morphological snake solver.
    levelset : array-like, optional
        If given, the levelset of the solver is initialized to this. If not
        given, the evolution will use the levelset already set in msnake.
    num_iters : int, optional
        The number of iterations.
    """
    from mayavi import mlab
    # import matplotlib.pyplot as ppl

    if levelset is not None:
        msnake.levelset = levelset

    fig = mlab.gcf()
    mlab.clf()
    src = mlab.pipeline.scalar_field(msnake.data)
    mlab.pipeline.image_plane_widget(
        src, plane_orientation='x_axes', colormap='gray')
    cnt = mlab.contour3d(msnake.levelset, contours=[0.5])

    @mlab.animate(ui=True)
    def anim():
        for i in range(num_iters):
            msnake.step()
            cnt.mlab_source.scalars = msnake.levelset
            yield

    anim()
    mlab.show()

    # Return the last levelset.
    return msnake.levelset
