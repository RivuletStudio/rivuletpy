import numpy as np
import skfmm
import math
from rivuletpy.utils.swc import flipswc
from scipy.ndimage.morphology import binary_dilation
from scipy.ndimage import generate_binary_structure
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD

from matplotlib import pyplot as plt

def riveal(img, swc, K=9, a=6, sample_rate=0.02):
    print('== oiginal image size: ', img.shape)
    # Pad the image and swc
    img = padimg(img, K*10)
    swc = padswc(swc, K*10)

    # plt.figure()
    # plt.imshow(img.max(axis=-1))
    # plt.figure()
    # plt.imshow(img.max(axis=-2))
    # plt.figure()
    # plt.imshow(img.max(axis=-3))

    print('== padded image size: ', img.shape)


    skimg = np.ones(img.shape)
    print('==Making skeleton image')
    # Make skeleton image
    for i in range(swc.shape[0]):
        node = [math.floor(n) for n in swc[i, 2:5]]
        skimg[node[0], node[1], node[2]] = 0

    # plt.figure()
    # plt.imshow(skimg.min(axis=-1))
    # plt.figure()
    # plt.imshow(skimg.min(axis=-2))
    # plt.figure()
    # plt.imshow(skimg.min(axis=-3))
    # plt.show()


    # Make the skeleton distance transform
    print('==Distance transform for swc')
    dm = math.floor(K/2)
    dt = skfmm.distance(skimg, dx=1)
    zeromask = dt >= dm
    dt = np.exp(a * (1 - dt / dm)) - 1
    dt[zeromask] = 0
    
    # Randomly sample 2.5D blocks from the include region 
    include_region = dt <= 1.5 * dm
    include_idx = np.argwhere(include_region)
    nidx = include_idx.shape[0]
    ntrain = np.floor(nidx * sample_rate)
    print('==ntrain:', ntrain)
    idx2train = include_idx[np.random.choice(nidx, ntrain),:]
    train_x = np.zeros((ntrain, 3, 2*K+1,2*K+1)) # Claim the memory for 2.5D blocks
    train_y = np.zeros((ntrain, 1)) # Claim the memory for 2.5D blocks

    for i in range(idx2train.shape[0]):
        bx, by, bz = idx2train[i, :]
        print('==bx:%d, by:%d, bz:%d' % (bx, by, bz))
        print(bx-K,bx+K+1, by-K,by+K+1, bz)
        print(bx-K,bx+K+1, by, bz-K,bz+K+1)
        print(bx, by-K,by+K+1, bz-K,bz+K+1)
        train_x[i, 0,:,:] = img[bx-K:bx+K+1, by-K:by+K+1, bz]
        train_x[i, 1,:,:] = img[bx-K:bx+K+1, by, bz-K:bz+K+1]
        train_x[i, 2,:,:] = img[bx, by-K:by+K+1, bz-K:bz+K+1]
        train_y[i] = dt[bx, by, bz] 

    # Build the CNN with keras+tensorflow
    model = traincnn(train_x, trainy, K)

    # Make the prediction within an area larger than the segmentation of the image
    bimg = img > 0
    bimg = binary_dilation(bimg)
    bimg = binary_dilation(bimg)
    bimg = binary_dilation(bimg)
    include_region = bimg > 0
    include_idx = np.argwhere(include_region)
    nidx = include_idx.shape[0]
    predict_x = zeros((nidx, 3, 2*K+1, 2*K+1))
    for i in range(include_idx.shape[0]):
        bx, by, bz = include_idx[i, :]
        predict_x[i, 0,:,:] = img[bx-K:bx+K+1, by-K:by+K+1, bz]
        predict_x[i, 1,:,:] = img[bx-K:bx+K+1, by, bz-K:bz+K+1]
        predict_x[i, 2,:,:] = img[bx, by-K:by+K+1, bz-K:bz+K+1]

    result_y = model.predict(predict_x, batch_size=64, verbose=0)
    img = img.fill(0)

    for i, y  in enumerate(result_y):
        bx, by, bz = include_idx[i, :]
        img[bx,by,bz] = y

    return img

def makecnn(K):
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, border_mode='same',
                        input_shape=X_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('linear'))
    return model


def traincnn(x, y, K):
    x = x.astype('float32')
    y = y.astype('float32')
    x /= x.max()
    y /= y.max()
    model = makecnn(K)
    model.compile(loss='mse',
                              optimizer='rmsprop',
                              metrics=['accuracy'])
    model.fit(x, y,
          batch_size=64,
          nb_epoch=10,
          # validation_data=(X_test, Y_test), # TODO
          shuffle=True)
    return model


def padimg(img, margin):
    pimg = np.zeros((img.shape[0]+2*margin, img.shape[1]+2*margin, img.shape[2]+2*margin))
    pimg[margin:margin+img.shape[0], margin:margin+img.shape[1], margin:margin+img.shape[2]] = img
    return pimg


def padswc(swc, margin):
    swc[:, 2:5] = swc[:, 2:5] + margin
    return swc
