import numpy as np


def fuzzy(img, level=128, p=2):
    '''
    Image auto thresholding with measure of fuzziness using the Yager's measure
    Implemented the algorithm following Eq.(11) in
    L. K. Huang and M. J. J. Wang, “Image thresholding by minimizing the
    measures of fuzziness,” Pattern Recognit., vol. 28, no. 1, pp. 41–51, 1995.
    img: a ND numpy matrix
    level: number of levels to partition the histogram
    p: order of metric  - 1 for Hamming metric - 2 for Euclidean metric
    '''

    assert (p == 1 or p == 2)

    high, low = img.max(), img.min(),
    step = (high - low) / level
    backweighted = foreweighted = 0
    nvox = img.size
    count = np.asarray([((img >= t - step / 2) & (img < t + step / 2)).sum()
                        for t in np.arange(low, high, step)])
    weighted = np.arange(low, high, step) * count
    backcount = np.asarray(
        [(img < t + step / 2).sum() for t in np.arange(low, high, step)])
    forecount = np.asarray([nvox] * level) - backcount
    backweighted = np.cumsum(weighted)
    # foreweighted = np.full((level,), backweighted[-1]) - backweighted
    foreweighted = np.cumsum(weighted[::-1])[::-1]
    muback = backweighted / backcount
    mufore = foreweighted / forecount
    yager = np.zeros((level, ))

    for i, t1 in enumerate(np.arange(low, high, step)):
        gsum = 0.
        for j, t2 in enumerate(np.arange(low, high, step)):
            mu = muback[i] if t2 <= t1 else mufore[i]
            C = np.abs(t1 - low) if t2 <= t1 else np.abs(high - t1)
            C = 1e-4 if C == 0 else C
            mux = 1. / (1. + np.abs(t2 - mu) / C)
            mux_reversed = 1. - mux
            gsum += np.abs(mux - mux_reversed)**p
        yager[i] = gsum**1 / p

    # The chosen threshold with least fuzziness
    return yager.argmin() * step + low, yager


def suppress(img, threshold):
    img[img <= threshold] = 0
    return img


def rescale(img, overwrite=False):
    '''
    Rescale image intensities linearly to 0~255
    '''
    if not overwrite:
        result = img.copy()
    else:
        result = img

    result = result.astype('float')
    result -= result.min()
    result /= result.max()
    result *= 255
    return result
