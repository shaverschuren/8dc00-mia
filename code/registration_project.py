"""
Registration project code.
"""

import numpy as np
import matplotlib.pyplot as plt
import registration as reg
from IPython.display import display, clear_output

# def fun(I, Im, x):
#     print(x)
#     f = reg.rigid_corr(I, Im, x)
#     C = f[0]
#     Im_t = f[1]
#     Th = f[2]
#     return C


def intensity_based_registration_demo():

    # read the fixed and moving images
    # change these in order to read different images
    I = plt.imread('../data/image_data/1_1_t1.tif')
    Im = plt.imread('../data/image_data/1_1_t1_d.tif')

    # initial values for the parameters
    # we start with the identity transformation
    # most likely you will not have to change these
    x = np.array([0., 0., 0.])

    # NOTE: for affine registration you have to initialize
    # more parameters and the scaling parameters should be
    # initialized to 1 instead of 0

    # the similarity function
    # this line of code in essence creates a version of rigid_corr()
    # in which the first two input parameters (fixed and moving image)
    # are fixed and the only remaining parameter is the vector x with the
    # parameters of the transformation

    # Function gave error due to passing of more than just the correlation
    # value (also transformed picture and transformation). This small change
    # solves this problem...
    fun = lambda x: (reg.rigid_corr(I, Im, x))[0]
    fun_full = lambda x: reg.rigid_corr(I, Im, x)

    # the learning rate
    mu = 0.001

    # number of iterations
    num_iter = 200

    iterations = np.arange(1, num_iter+1)
    similarity = np.full((num_iter, 1), np.nan)

    fig = plt.figure(figsize=(14,6))

    # fixed and moving image, and parameters
    ax1 = fig.add_subplot(121)

    # fixed image
    im1 = ax1.imshow(I)
    # moving image
    im2 = ax1.imshow(I, alpha=0.7)
    # parameters
    txt = ax1.text(0.3, 0.95,
        np.array2string(x, precision=5, floatmode='fixed'),
        bbox={'facecolor': 'white', 'alpha': 1, 'pad': 10},
        transform=ax1.transAxes)

    # 'learning' curve
    ax2 = fig.add_subplot(122, xlim=(0, num_iter), ylim=(0, 1))

    learning_curve, = ax2.plot(iterations, similarity, lw=2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Similarity')
    ax2.grid()

    # perform 'num_iter' gradient ascent updates
    for k in np.arange(num_iter):

        # gradient ascent
        g = reg.ngradient(fun, x)
        x += g*mu

        # for visualization of the result
        S, Im_t, _ = fun_full(x)

        clear_output(wait = True)

        # update moving image and parameters
        im2.set_data(Im_t)
        txt.set_text(np.array2string(x, precision=5, floatmode='fixed'))

        # update 'learning' curve
        similarity[k] = S
        learning_curve.set_ydata(similarity)

        display(fig)


def intensity_based_registration(I_path, Im_path, r_a_switch=0):

    # r_a_switch: 0 --> rigid (default)
    #             1 --> affine
    assert r_a_switch == 0 or r_a_switch == 1, "Error: input parameter r_a_switch must be either 0 or 1.. "

    # read the fixed and moving images
    # change these in order to read different images
    I = plt.imread(I_path)
    Im = plt.imread(Im_path)

    # initial values for the parameters
    if r_a_switch == 0:
        x = np.array([0., 0., 0.]) # rotation,transx,transy
    elif r_a_switch == 1:
        x = np.array([0., 1., 1.,0.,0.,0.,0.]) # rotation,scalex,scaley,shearx,sheary,transx,transy
    else:
        print("ERROR.. r_a_switch must be either 0 or 1")

    # the similarity function
    # this line of code in essence creates a version of rigid_corr()
    # in which the first two input parameters (fixed and moving image)
    # are fixed and the only remaining parameter is the vector x with the
    # parameters of the transformation
    if r_a_switch == 0:
        fun = lambda x: (reg.rigid_corr(I, Im, x))[0]
        fun_full = lambda x: reg.rigid_corr(I, Im, x)
    elif r_a_switch == 1:
        fun = lambda x: (reg.affine_corr(I, Im, x))[0]
        fun_full = lambda x: reg.affine_corr(I, Im, x)
    else:
        print("ERROR.. r_a_switch must be either 0 or 1")

    # the learning rate
    mu = 0.001

    # number of iterations
    num_iter = 200

    iterations = np.arange(1, num_iter+1)
    similarity = np.full((num_iter, 1), np.nan)

    fig = plt.figure(figsize=(14,6))

    # fixed and moving image, and parameters
    ax1 = fig.add_subplot(121)

    # fixed image
    im1 = ax1.imshow(I)
    # moving image
    im2 = ax1.imshow(I, alpha=0.7)
    # parameters
    txt = ax1.text(0.3, 0.95,
        np.array2string(x, precision=5, floatmode='fixed'),
        bbox={'facecolor': 'white', 'alpha': 1, 'pad': 10},
        transform=ax1.transAxes)

    # 'learning' curve
    ax2 = fig.add_subplot(122, xlim=(0, num_iter), ylim=(0, 1))

    learning_curve, = ax2.plot(iterations, similarity, lw=2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Similarity')
    ax2.grid()

    # perform 'num_iter' gradient ascent updates
    for k in np.arange(num_iter):

        # gradient ascent
        g = reg.ngradient(fun, x)
        x += g*mu

        # for visualization of the result
        S, Im_t, _ = fun_full(x)

        clear_output(wait = True)

        # update moving image and parameters
        im2.set_data(Im_t)
        txt.set_text(np.array2string(x, precision=5, floatmode='fixed'))

        # update 'learning' curve
        similarity[k] = S
        learning_curve.set_ydata(similarity)

        display(fig)
