import numpy as np
import math
import skfmm
from tqdm import tqdm
from scipy.ndimage.morphology import binary_dilation
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.noise import GaussianDropout, GaussianNoise
from keras.layers.advanced_activations import SReLU


def riveal(img, swc, K=9, nsample=8e4, epoch=20):
    print('-- oiginal image size: ', img.shape)
    K = math.floor(K)  # Make sure K is integer to avoid warnings
    # Pad the image and swc
    margin = 3 * K
    img = padimg(img, margin)
    swc = padswc(swc, margin)

    # Make the skeleton distance transform
    print('-- Distance transform for swc...')
    dt, foreground_region = make_skdt(img.shape, swc, K)

    # Normalise data
    # img = standardise(img)
    # dt = standardise(dt)
    img /= img.max()
    # dt /= dt.max()

    # Make the confident region
    print('==swc shape:', swc.shape)
    print('-- Making the confidence regions...(1/4)')
    high_conf_region = make_conf_region(
        img.shape, swc, K, low_conf=0.5, high_conf=1.)
    # print('-- Making the confidence regions...(2/4)')
    # mid_conf_region = make_conf_region(img.shape, swc, K,
    #                                    low_conf=0.25, high_conf=0.5)
    print('-- Making the confidence regions...(3/4)')
    low_conf_region = make_conf_region(
        img.shape, swc, K, low_conf=0., high_conf=0.25)

    # # Fill only the central part of background region
    print('-- Making the confidence regions...(4/4)')
    background_region = np.zeros(img.shape)
    bg = np.logical_not(foreground_region)
    bg = np.logical_and(foreground_region, img > 0)
    for i in range(3):
        bg = binary_dilation(bg)
    background_region[margin:-margin, margin:-margin, margin:-margin] = bg[
        margin:-margin, margin:-margin, margin:-margin]

    from matplotlib import pyplot as plt
    plt.subplot(3, 1, 1)
    plt.imshow(high_conf_region.max(axis=-1))
    plt.title('high conf')
    plt.subplot(3, 1, 2)
    plt.imshow(low_conf_region.max(axis=-1))
    plt.title('low conf')
    plt.subplot(3, 1, 3)
    plt.imshow(background_region.max(axis=-1))
    plt.title('bg')
    plt.show()

    # Randomly sample 2.5D blocks from the include region
    print('-- Sampling blocks')
    x1, y1 = sample_block(img, dt, high_conf_region, K,
                          math.ceil(nsample * 0.75))
    # x2, y2 = sample_block(img, dt, mid_conf_region,
    #                       K, math.ceil(nsample * 0.2))
    x3, y3 = sample_block(img, dt, low_conf_region, K,
                          math.ceil(nsample * 0.1))
    y3.fill(0.)
    x4, y4 = sample_block(img, dt, background_region, K,
                          math.ceil(nsample * 0.15))
    y4.fill(0.)
    train_x = np.vstack((x1, x3, x4))
    train_y = np.vstack((y1, y3, y4))

    # Build the CNN with keras+tensorflow
    print('--Training CNN...')
    model = traincnn(train_x, train_y, K, epoch)

    # Make the prediction within an area larger than
    # the segmentation of the image
    print('-- Predicting...')
    bimg = img > 0
    for i in range(6):
        bimg = binary_dilation(bimg)
    include_region = bimg > 0
    include_idx = np.argwhere(include_region)
    nidx = include_idx.shape[0]

    predict_x = np.zeros((nsample, 2 * K + 1, 2 * K + 1, 3))
    rest = nidx
    resultimg = np.zeros(img.shape)
    pbar = tqdm(total=nidx)

    # Predict every batch of blocks
    while rest > 0:
        startidx = -rest
        endidx = -rest + nsample if -rest + nsample < nidx else nidx
        rest -= nsample

        # Write the value to each include voxel
        for i, gidx in enumerate(range(int(startidx), int(endidx))):
            bx, by, bz = include_idx[gidx, :]
            predict_x[i, :, :, 0] = img[bx - K:bx + K + 1, by - K:by + K + 1,
                                        bz]
            predict_x[i, :, :, 1] = img[bx - K:bx + K + 1, by, bz - K:bz + K +
                                        1]
            predict_x[i, :, :, 2] = img[bx, by - K:by + K + 1, bz - K:bz + K +
                                        1]

        pd = model.predict(predict_x, batch_size=64, verbose=0).flatten()

        for i, gidx in enumerate(range(int(startidx), int(endidx))):
            bx, by, bz = include_idx[gidx, :]
            resultimg[bx, by, bz] = pd[i]
        pbar.update(nsample)

    resultimg = unpadimg(resultimg, margin)

    return resultimg


def standardise(img, zeromean=True):
    img = (img - img.mean()) / img.std()
    return img


def constrain_range(min, max, minlimit, maxlimit):
    return list(
        range(min if min > minlimit else minlimit, max
              if max < maxlimit else maxlimit))


def sample_block(img, dt, include_region, K, nsample):
    include_idx = np.argwhere(include_region)
    nidx = include_idx.shape[0]
    nsample = nidx if nsample > nidx else nsample
    idx2train = include_idx[np.random.choice(nidx, nsample), :]

    # Claim the memory for 2.5D blocks
    x = np.zeros((nsample, 2 * K + 1, 2 * K + 1, 3))
    y = np.zeros((nsample, 1))  # Claim the memory for 2.5D blocks

    for i in range(idx2train.shape[0]):
        bx, by, bz = idx2train[i, :]
        x[i, :, :, 0] = img[bx - K:bx + K + 1, by - K:by + K + 1, bz]
        x[i, :, :, 1] = img[bx - K:bx + K + 1, by, bz - K:bz + K + 1]
        x[i, :, :, 2] = img[bx, by - K:by + K + 1, bz - K:bz + K + 1]
        y[i] = dt[bx, by, bz]

    return x, y


def make_conf_region(imshape, swc, K, low_conf=0.0, high_conf=1.0):
    if low_conf != 0.0 or high_conf != 1.0:
        confswc = np.vstack((swc[np.logical_and(swc[:, 7] >= low_conf,
                                                swc[:, 7] <= high_conf), :]))

    region = np.zeros(imshape)
    r = math.ceil(K * 0.75)
    for i in range(confswc.shape[0]):
        node = confswc[i, :]
        n = [math.floor(n) for n in node[2:5]]
        rg1 = constrain_range(n[0] - r, n[0] + r + 1, 0, imshape[0])
        rg2 = constrain_range(n[1] - r, n[1] + r + 1, 0, imshape[1])

        rg3 = constrain_range(n[2] - r, n[2] + r + 1, 0, imshape[2])
        X, Y, Z = np.meshgrid(rg1, rg2, rg3)

        # Skip if any node has empty box
        if len(X) == 0 or len(Y) == 0 or len(Z) == 0:
            continue
        region[X, Y, Z] = 1
    # _, region = make_skdt(imshape, confswc, K)
    return region


def make_skdt(imshape, swc, K, a=6):
    skimg = make_sk_img(imshape, swc)
    dm = math.floor(K / 2)
    dt = skfmm.distance(skimg, dx=1)
    include_region = dt <= 1.5 * dm
    zeromask = dt >= dm
    dt = np.exp(a * (1 - dt / dm)) - 1
    dt[zeromask] = 0

    return dt, include_region


def makecnn(in_shape, K):
    model = Sequential()
    model.add(
        Convolution2D(
            32, 3, 3, border_mode='same', input_shape=in_shape[1:]))
    model.add(SReLU())
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='tf'))
    model.add(GaussianNoise(1))
    model.add(GaussianDropout(0.4))
    model.add(Convolution2D(32, 3, 3, border_mode='same'))
    model.add(SReLU())
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='tf'))
    model.add(GaussianNoise(1))
    model.add(GaussianDropout(0.4))
    model.add(Flatten())
    model.add(Dense(64))
    model.add(SReLU())
    model.add(Dense(64))
    # model.add(SReLU())
    model.add(Dense(1))
    model.add(Activation('linear'))
    return model


def traincnn(x, y, K, epoch):
    x = x.astype('float32')
    y = y.astype('float32')
    x /= x.max()
    y /= y.max()
    model = makecnn(x.shape, K)
    model.compile(loss='mse', optimizer='rmsprop')
    model.fit(x,
              y,
              batch_size=64,
              nb_epoch=epoch,
              validation_split=0.15,
              shuffle=True)
    return model


def make_sk_img(imshape, swc):
    skimg = np.ones(imshape)
    for i in range(swc.shape[0]):
        node = [math.floor(n) for n in swc[i, 2:5]]
        skimg[node[0], node[1], node[2]] = 0
    return skimg


def padimg(img, margin):
    pimg = np.zeros((img.shape[0] + 2 * margin, img.shape[1] + 2 * margin,
                     img.shape[2] + 2 * margin))
    pimg[margin:margin + img.shape[0], margin:margin + img.shape[1], margin:
         margin + img.shape[2]] = img
    return pimg


def unpadimg(img, margin):
    pimg = np.zeros((img.shape[0] - 2 * margin, img.shape[1] - 2 * margin,
                     img.shape[2] - 2 * margin))
    pimg = img[margin:margin + img.shape[0], margin:margin + img.shape[1],
               margin:margin + img.shape[2]]
    return pimg


def padswc(swc, margin):
    swc[:, 2:5] = swc[:, 2:5] + margin
    return swc
