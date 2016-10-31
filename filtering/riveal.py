import numpy as np
import math
import skfmm
from tqdm import tqdm
from scipy.ndimage.morphology import binary_dilation
from scipy.ndimage import generate_binary_structure
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from rivuletpy.utils.swc import flipswc

def riveal(img, swc, K=9, a=6, nsample=5e4, epoch=20):
    print('== oiginal image size: ', img.shape)
    # Pad the image and swc
    margin = 2 * K
    img = padimg(img, margin)
    swc = padswc(swc, margin)

    skimg = np.ones(img.shape)
    print('==Making skeleton image')
    # Make skeleton image
    for i in range(swc.shape[0]):
        node = [math.floor(n) for n in swc[i, 2:5]]
        skimg[node[0], node[1], node[2]] = 0

    # Make the skeleton distance transform
    print('==Distance transform for swc')
    dm = math.floor(K/2)
    dt = skfmm.distance(skimg, dx=1)
    include_region = dt <= 1.5 * dm
    zeromask = dt >= dm
    dt = np.exp(a * (1 - dt / dm)) - 1
    dt[zeromask] = 0

    # Randomly sample 2.5D blocks from the include region 
    include_idx = np.argwhere(include_region)
    nidx = include_idx.shape[0]
    idx2train = include_idx[np.random.choice(nidx, nsample),:]
    train_x = np.zeros((nsample, margin+1,margin+1, 3)) # Claim the memory for 2.5D blocks
    train_y = np.zeros((nsample, 1)) # Claim the memory for 2.5D blocks

    # Normalise data
    img /= img.max()
    dt /= dt.max()

    for i in range(idx2train.shape[0]):
        bx, by, bz = idx2train[i, :]
        train_x[i, :,:, 0] = img[bx-K:bx+K+1, by-K:by+K+1, bz]
        train_x[i, :,:, 1] = img[bx-K:bx+K+1, by, bz-K:bz+K+1]
        train_x[i, :,:, 2] = img[bx, by-K:by+K+1, bz-K:bz+K+1]
        train_y[i] = dt[bx, by, bz] 

    # Build the CNN with keras+tensorflow
    print('==Training CNN...')
    model = traincnn(train_x, train_y, K, epoch)

    # Make the prediction within an area larger than the segmentation of the image
    print('==Making predictions...')
    bimg = img > 0
    bimg = binary_dilation(bimg)
    bimg = binary_dilation(bimg)
    bimg = binary_dilation(bimg)
    include_region = bimg > 0
    include_idx = np.argwhere(include_region)
    nidx = include_idx.shape[0]

    predict_x = np.zeros((nsample, margin+1, margin+1, 3))
    rest = nidx
    resultimg = np.zeros(img.shape)
    pbar = tqdm(total=nidx)
    while rest > 0:
        startidx = -rest
        endidx = -rest+nsample if -rest+nsample < nidx else nidx
        rest -= nsample
        # print('start:', startidx, 'end:', endidx)

        for i, gidx in enumerate(range(int(startidx), int(endidx))):
            bx, by, bz = include_idx[gidx, :]
            predict_x[i,:,:,0] = img[bx-K:bx+K+1, by-K:by+K+1, bz]
            predict_x[i,:,:,1] = img[bx-K:bx+K+1, by, bz-K:bz+K+1]
            predict_x[i,:,:,2] = img[bx, by-K:by+K+1, bz-K:bz+K+1]

        pd = model.predict(predict_x, batch_size=64, verbose=0).flatten()

        for i, gidx in enumerate(range(int(startidx), int(endidx))):
            bx, by, bz = include_idx[gidx, :]
            resultimg[bx,by,bz] = pd[i]
        pbar.update(nsample)

    resultimg = unpadimg(resultimg, margin)

    return resultimg


def makecnn(in_shape, K):
    print(in_shape)
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, border_mode='same',
                        input_shape=in_shape[1:]))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='tf'))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('linear'))
    return model


def traincnn(x, y, K, epoch):
    x = x.astype('float32')
    y = y.astype('float32')
    x /= x.max()
    y /= y.max()
    model = makecnn(x.shape, K)
    model.compile(loss='mse',
                              optimizer='rmsprop',
                              metrics=['accuracy'])
    model.fit(x, y,
          batch_size=64,
          nb_epoch=epoch,
          # validation_data=(X_test, Y_test), # TODO
          shuffle=True)
    return model


def padimg(img, margin):
    pimg = np.zeros((img.shape[0]+2*margin, img.shape[1]+2*margin, img.shape[2]+2*margin))
    pimg[margin:margin+img.shape[0], margin:margin+img.shape[1], margin:margin+img.shape[2]] = img
    return pimg


def unpadimg(img, margin):
    pimg = np.zeros((img.shape[0]-2*margin, img.shape[1]-2*margin, img.shape[2]-2*margin))
    pimg = img[margin:margin+img.shape[0], margin:margin+img.shape[1], margin:margin+img.shape[2]]
    return pimg


def padswc(swc, margin):
    swc[:, 2:5] = swc[:, 2:5] + margin
    return swc
