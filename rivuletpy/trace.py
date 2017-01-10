import math
from tqdm import tqdm
import numpy as np
import skfmm
import msfm
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage.morphology import binary_dilation
from filtering.morphology import ssm
from skimage.filters import threshold_otsu
from .soma import Soma
from .swc import SWC


class Tracer(object):

    def __init__(self):
        pass

    def reset(self):
        pass

    def trace(self):
        pass


class R2Tracer(Tracer):

    def __init__(self, quality=False, silent=False, speed='dt', clean=False):
        self._quality = quality
        self._bimg = None
        self._dilated_bimg = None
        self._bsum = 0  # For counting the covered foreground
        self._bb = None  # For making the erasing contour
        self._t = None  # Original timemap
        self._tt = None  # The copy of the timemap
        self._grad = None
        self._coverage = 0.
        self._soma = None  # soma
        self._silent = silent  # Disable all console outputs
        # Tracing stops when 98% of the foreground has been covered
        self._target_coverage = 0.98
        self._cover_ctr_old = 0.
        self._cover_ctr_new = 0.
        # The type of speed image to use. Options are ['dt', 'ssm']
        self._speed = speed
        self._erase_ratio = 1.7 if self._speed == 'ssm' else 1.5
        # Whether the unconnected branches will be discarded
        self._clean = clean
        self._eps = 1e-5


    def trace(self, img, threshold):
        '''
        The main entry for Rivulet2
        '''
        self._bimg = (img > threshold).astype('int')  # Segment image
        if not self._silent: print('(1) -- Detecting Soma...', end='')
        self._soma = Soma()
        self._soma.detect(self._bimg, not self._quality, self._silent)
        self._prep()

        # Iterative Back Tracking with Erasing
        if not self._silent:
            print('(5) --Start Backtracking...')
        swc = self._iterative_backtrack()

        if self._clean:
            swc.prune()

        return swc, self._soma

    def _prep(self):
        self._nforeground = self._bimg.sum()        
        # Dilate bimg to make it less strict for the big gap criteria
        # It is needed since sometimes the tracing goes along the
        # boundary of the thin fibre in the binary img
        self._dilated_bimg = binary_dilation(self._bimg)

        if not self._silent:
            print('(2) --Boundary DT...')
        self._make_dt()
        if not self._silent:
            print('(3) --Fast Marching with %s quality...' % ('high' if self._quality else 'low'))
        self._fast_marching()
        if not self._silent:
            print('(4) --Compute Gradients...')
        self._make_grad()

        # Make copy of the timemap
        self._tt = self._t.copy()
        self._tt[self._bimg <= 0] = -2

        # Label all voxels of soma with -3
        self._tt[self._soma.mask > 0] = -3

        # For making a large tube to contain the last traced branch
        self._bb = np.zeros(shape=self._tt.shape)

    def _update_coverage(self):
        self._cover_ctr_new = np.logical_and(self._tt < 0, self._bimg > 0).sum()

        self._coverage = self._cover_ctr_new / self._nforeground
        if not self._silent: self._pbar.update(self._cover_ctr_new - self._cover_ctr_old)
        self._cover_ctr_old = self._cover_ctr_new


    def _make_grad(self):
        # Get the gradient of the Time-crossing map
        dx, dy, dz = self._dist_gradient()
        standard_grid = (np.arange(self._t.shape[0]), np.arange(self._t.shape[1]),
                         np.arange(self._t.shape[2]))
        self._grad = (RegularGridInterpolator(standard_grid, dx),
                      RegularGridInterpolator(standard_grid, dy),
                      RegularGridInterpolator(standard_grid, dz))


    def _make_dt(self):
        '''
        Make the distance transform according to the speed type
        '''
        self._dt = skfmm.distance(self._bimg, dx=5e-2)  # Boundary DT

        if self._speed == 'ssm':
            if not self._silence:
                print('--SSM with GVF...')
            self._dt = ssm(self._dt, anisotropic=True, iterations=40)
            img = self._dt > threshold_otsu(self._dt)
            self._dt = skfmm.distance(img, dx=5e-2)
            self._dt = skfmm.distance(np.logical_not(self._dt), dx=5e-3)
            self._dt[self._dt > 0.04] = 0.04
            self._dt = self._dt.max() - self._dt

    def _fast_marching(self):
        speed = self._make_speed(self._dt)
        # # Fast Marching
        if self._quality:
            # if not self._silent: print('--MSFM...')
            self._t = msfm.run(speed, self._bimg.copy().astype('int64'), self._soma.centroid, True, True)
        else:
            # if not self._silent: print('--FM...')
            marchmap = np.ones(self._bimg.shape)
            marchmap[self._soma.centroid[0], self._soma.centroid[1], self._soma.centroid[2]] = -1
            self._t = skfmm.travel_time(marchmap, speed, dx=5e-3)

    def _make_speed(self, dt):
        F = dt**4
        F[F <= 0] = 1e-10
        return F

    def _dist_gradient(self):
        fx = np.zeros(shape=self._t.shape)
        fy = np.zeros(shape=self._t.shape)
        fz = np.zeros(shape=self._t.shape)

        J = np.zeros(shape=[s + 2 for s in self._t.shape])  # Padded Image
        J[:, :, :] = self._t.max()
        J[1:-1, 1:-1, 1:-1] = self._t
        Ne = [[-1, -1, -1], [-1, -1, 0], [-1, -1, 1], [-1, 0, -1], [-1, 0, 0],
              [-1, 0, 1], [-1, 1, -1], [-1, 1, 0], [-1, 1, 1], [0, -1, -1],
              [0, -1, 0], [0, -1, 1], [0, 0, -1], [0, 0, 1], [0, 1, -1],
              [0, 1, 0], [0, 1, 1], [1, -1, -1], [1, -1, 0], [1, -1, 1],
              [1, 0, -1], [1, 0, 0], [1, 0, 1], [1, 1, -1], [1, 1, 0], [1, 1, 1]]

        for n in Ne:
            In = J[1 + n[0]:J.shape[0] - 1 + n[0], 1 + n[1]:J.shape[1] - 1 + n[1],
                   1 + n[2]:J.shape[2] - 1 + n[2]]
            check = In < self._t
            self._t[check] = In[check]
            D = np.divide(n, np.linalg.norm(n))
            fx[check] = D[0]
            fy[check] = D[1]
            fz[check] = D[2]
        return -fx, -fy, -fz

    def _step(self, branch):
        # RK4 Walk for one step
        p = rk4(branch.pts[-1], self._grad, self._t, 1)
        branch.update(p, self._bimg, self._dilated_bimg)

    def _erase(self, branch):
        # Erase it from the timemap
        for i in range(len(branch.pts)):
            n = [math.floor(n) for n in branch.pts[i]]
            r = 1 if branch.radius[i] < 1 else branch.radius[i]

            # To make sure all the foreground voxels are included in bb
            r = math.ceil(r * self._erase_ratio)
            X, Y, Z = np.meshgrid(
                        constrain_range(n[0] - r, n[0] + r + 1, 0, self._tt.shape[0]),
                        constrain_range(n[1] - r, n[1] + r + 1, 0, self._tt.shape[1]),
                        constrain_range(n[2] - r, n[2] + r + 1, 0, self._tt.shape[2]))
            self._bb[X, Y, Z] = 1

        startidx, endidx = [math.floor(p) for p in branch.pts[0]], [math.floor(p) for p in branch.pts[-1]]

        if len(branch.pts) > 5 and self._t[endidx[0], endidx[1], endidx[2]] < self._t[
                startidx[0], startidx[1], startidx[2]]:
            erase_region = np.logical_and(
                self._t[endidx[0], endidx[1], endidx[2]] <= self._t,
                self._t <= self._t[startidx[0], startidx[1], startidx[2]])
            erase_region = np.logical_and(self._bb, erase_region)
        else:
            erase_region = self._bb.astype('bool')

        if np.count_nonzero(erase_region) > 0:
            self._tt[erase_region] = -2 if branch.low_conf else -1
        self._bb.fill(0)

    def _iterative_backtrack(self):

        # Initialise swc with the soma centroid
        swc = SWC(self._soma)
        swc.add(np.reshape(
            np.asarray([
                0, 1, self._soma.centroid[0], self._soma.centroid[1], self._soma.centroid[2],
                self._soma.radius, -1, 1.
            ]), (1, 8)))
        

        if not self._silent:
            self._pbar = tqdm(total=math.floor(self._nforeground * self._target_coverage))

        # Loop for all branches
        while self._coverage < self._target_coverage:
            self._update_coverage()
            # Find the geodesic furthest point on foreground time-crossing-map
            srcpt = np.asarray(np.unravel_index(self._tt.argmax(), self._tt.shape)).astype('float64')
            branch = R2Branch()
            branch.add(srcpt, 1., 1.)

            # Erase the source point just in case
            self._tt[math.floor(srcpt[0]), math.floor(srcpt[1]), math.floor(srcpt[2])] = -2
            keep = True

            # Loop for 1 back-tracking iteration
            while True:
                self._step(branch)
                head = branch.pts[-1]
                tt_head = self._tt[math.floor(head[0]), math.floor(head[1]), math.floor(head[2])]

                # 1. Check out of bound
                if not inbound(head, self._bimg.shape):
                    branch.slice(0, -1)
                    break

                # 2. Check for the large gap criterion
                if branch.gap > np.asarray(branch.radius).mean() * 8:
                    break
                else:
                    branch.reset_gap()

                # 3. Check if Soma has been reached
                if tt_head  == -3:
                    keep = True if branch.branchlen > self._soma.radius * 3 else False
                    branch.reached_soma = True
                    break

                # 4. Check if not moved for 15 iterations
                if branch.is_stucked():
                    break

                # 5. Check for low online confidence 
                if branch.low_conf:
                    keep = False
                    break

                # 6. Check for branch merge
                # Consider reaches previous explored area traced with branch
                # Note: when the area was traced due to noise points
                # (erased with -2), not considered as 'reached'
                if tt_head == -1:
                    branch.touched = True
                    if swc.size() == 1:
                        break

                    matched, matched_idx = swc.match(head, branch.radius[-1])
                    if matched > 0:
                        branch.touch_idx = matched_idx
                        break

                    if branch.steps_after_reach > 200:
                        break
            
            self._erase(branch)

            # Add to SWC if it was decided to be kept
            if keep:
                pidx = None 
                if branch.reached_soma:
                    pidx = 0;
                elif branch.touch_idx >= 0:
                    pidx = branch.touch_idx
                swc.add_branch(branch, pidx)
        return swc

class Branch(object):
    def __init__(self):
        self.pts = []
        self.radius = []

class R2Branch(Branch):
    def __init__(self):
        self.pts = []
        self.conf = []
        self.radius = []
        self.steps_after_reach = 0
        self.low_conf = False
        self.touch_idx = -2
        self.reached_soma = False
        self.branchlen = 0
        self.gap = 0
        self.online_voxsum = 0
        self.stepsz = 0
        self.touched = False

        self.ma_short = -1
        self.ma_long = -1
        self.ma_short_window = 4
        self.ma_long_window = 10
        self.in_valley = False


    def add(self, pt, conf, radius):
        self.pts.append(pt)
        self.conf.append(conf)
        self.radius.append(radius)

    def is_stucked(self):
        if self.stepsz == 0:
            return True

        if len(self.pts) > 15:
            if np.linalg.norm(np.asarray(self.pts[-1]) - np.asarray(self.pts[-15])) < 1:
                return True
            else:
                return False
        else:
            return False

    def reset_gap(self):
        self.gap = 0

    def update(self, pt, bimg, dilated_bimg):
        eps = 1e-5
        head = self.pts[-1]
        velocity = np.asarray(pt) - np.asarray(head)
        self.stepsz = np.linalg.norm(velocity)
        self.branchlen += self.stepsz
        b = dilated_bimg[math.floor(pt[0]), math.floor(pt[1]), math.floor(pt[2])]
        if b > 0:
            self.gap += self.stepsz
        
        self.online_voxsum += b
        oc = self.online_voxsum / (len(self.pts) + 1)
        self.update_ma(oc)

         # We are stepping in a valley
        if (self.ma_short < self.ma_long - eps and
                oc < 0.5 and not self.in_valley):
            self.in_valley = True
        
        # Cut at the valley
        if self.in_valley and self.ma_short > self.ma_long:
            valleyidx = np.asarray(self.conf).argmin()
            # Only cut if the valley confidence is below 0.5
            if self.conf[valleyidx] < 0.5:
                self.slice(0, valleyidx)
                self.low_conf = True
            else:
                in_valley = False

        if oc <= 0.2:
            self.low_conf = True

        if self.touched:
            self.steps_after_reach += 1

        r = estimate_radius(pt, bimg)
        self.add(pt, oc, r)

    def update_ma(self, oc):
        if len(self.pts) > self.ma_long_window:
            if self.ma_short == -1:
                self.ma_short = oc
            else:
                self.ma_short = exponential_moving_average(
                        oc, self.ma_short, self.ma_short_window
                        if len(self.pts) >= self.ma_short_window else len(self.pts))
            if self.ma_long == -1:
                self.ma_long = oc
            else:
                self.ma_long = exponential_moving_average(
                    oc, self.ma_long, self.ma_long_window
                    if len(self.pts) >= self.ma_long_window else len(self.pts))

    def slice(self, start, end):
        self.pts = self.pts[start: end]
        self.radius = self.radius[start: end]
        self.conf = self.conf[start: end]


def estimate_radius(pt, bimg):
    r = 0
    x = math.floor(pt[0])
    y = math.floor(pt[1])
    z = math.floor(pt[2])

    while True:
        r += 1
        try:
            if bimg[max(x - r, 0):min(x + r + 1, bimg.shape[0]), max(y - r, 0):
                    min(y + r + 1, bimg.shape[1]), max(z - r, 0):min(
                        z + r + 1, bimg.shape[2])].sum() / (2 * r + 1)**3 < .6:
                break
        except IndexError:
            break

    return r



def exponential_moving_average(p, ema, n):
    '''
    The exponential moving average (EMA) traditionally
    used in analysing stock market.
    EMA_{i+1} = (p * \alpha) + (EMA_{i} * (1 - \alpha))
    where p is the new value; EMA_{i} is the last ema value;
    n is the time period; \alpha=2/(1+n) is the smoothing factor.

    ---------------------------------------------
    Parameters:
    p: The new value in the sequence
    ema: the last EMA value
    n: The period window size
    '''

    alpha = 2 / (1 + n)
    return p * alpha + ema * (1 - alpha)


def rk4(srcpt, ginterp, t, stepsize):
    # Compute K1
    k1 = np.asarray([g(srcpt)[0] for g in ginterp])
    k1 *= stepsize / max(np.linalg.norm(k1), 1.)
    tp = srcpt - 0.5 * k1  # Position of temporary point
    if not inbound(tp, t.shape):
        return srcpt

    # Compute K2
    k2 = np.asarray([g(tp)[0] for g in ginterp])
    k2 *= stepsize / max(np.linalg.norm(k2), 1.)
    tp = srcpt - 0.5 * k2  # Position of temporary point
    if not inbound(tp, t.shape):
        return srcpt

    # Compute K3
    k3 = np.asarray([g(tp)[0] for g in ginterp])
    k3 *= stepsize / max(np.linalg.norm(k3), 1.)
    tp = srcpt - k3  # Position of temporary point
    if not inbound(tp, t.shape):
        return srcpt

    # Compute K4
    k4 = np.asarray([g(tp)[0] for g in ginterp])
    k4 *= stepsize / max(np.linalg.norm(k4), 1.)

    return srcpt - (k1 + k2 * 2 + k3 * 2 + k4) / 6.0  # Compute final point


def inbound(pt, shape):
    return all([True if 0 <= p <= s - 1 else False for p, s in zip(pt, shape)])


def constrain_range(min, max, minlimit, maxlimit):
    return list(
        range(min if min > minlimit else minlimit, max
              if max < maxlimit else maxlimit))
