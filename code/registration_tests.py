"""
Test code for registration.
"""

import numpy as np
import matplotlib.pyplot as plt
import registration as reg
import registration_util as util
from IPython.display import display, clear_output
from math import *


# SECTION 1. Geometrical transformations

def transforms_test():

    X = util.test_object(1)

    X_rot = reg.rotate(3*np.pi/4).dot(X)
    X_shear = reg.shear(0.1, 0.2).dot(X)
    X_reflect = reg.reflect(-1, -1).dot(X)

    fig = plt.figure(figsize=(12,5))
    ax1 = fig.add_subplot(141, xlim=(-4,4), ylim=(-4,4))
    ax2 = fig.add_subplot(142, xlim=(-4,4), ylim=(-4,4))
    ax3 = fig.add_subplot(143, xlim=(-4,4), ylim=(-4,4))
    ax4 = fig.add_subplot(144, xlim=(-4,4), ylim=(-4,4))

    util.plot_object(ax1, X)
    util.plot_object(ax2, X_rot)
    util.plot_object(ax3, X_shear)
    util.plot_object(ax4, X_reflect)

    ax1.set_title('Original')
    ax2.set_title('Rotation')
    ax3.set_title('Shear')
    ax4.set_title('Reflection')

    ax1.grid()
    ax2.grid()
    ax3.grid()
    ax4.grid()


def combining_transforms():

    X = util.test_object(1)

    X_1 = X
    X_2 = reg.rotate(3*np.pi/4).dot(X)
    X_3 = reg.reflect(-1, 1).dot(reg.rotate(3 * np.pi / 4).dot(X))
    X_4 = reg.shear(0.1,0.2).dot(reg.reflect(-1, 1).dot(reg.rotate(3 * np.pi / 4).dot(X)))

    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(141, xlim=(-4, 4), ylim=(-4, 4))
    ax2 = fig.add_subplot(142, xlim=(-4, 4), ylim=(-4, 4))
    ax3 = fig.add_subplot(143, xlim=(-4, 4), ylim=(-4, 4))
    ax4 = fig.add_subplot(144, xlim=(-4, 4), ylim=(-4, 4))

    util.plot_object(ax1, X_1)
    util.plot_object(ax2, X_2)
    util.plot_object(ax3, X_3)
    util.plot_object(ax4, X_4)

    ax1.set_title('Original')
    ax2.set_title('Then rotate')
    ax3.set_title('Now reflect over x')
    ax4.set_title('Lastly, shear')

    for ax_obj in [ax1,ax2,ax3,ax4]:
        ax_obj.grid()

def t2h_test():

    X = util.test_object(1)
    Xh = util.c2h(X)

    # translation vector
    t = np.array([10, 20])

    # rotation matrix
    T_rot = reg.rotate(np.pi/4)

    Th = util.t2h(T_rot, t)

    X_rot_tran = Th.dot(Xh)

    fig = plt.figure(figsize=(5,5))
    ax1 = fig.add_subplot(111)
    util.plot_object(ax1, X)
    util.plot_object(ax1, X_rot_tran)
    ax1.grid()


def arbitrary_rotation():

    X = util.test_object(1)
    Xh = util.c2h(X)

    #Define rotation angle 45degrees
    phi = np.pi/4

    #Define seperate homogenous transformation matrices
    T_1 = np.array([[1,0,-X[0,0]],[0,1,-X[1,0]],[0,0,1]])
    T_rot = np.array([[cos(phi),-sin(phi),0],[sin(phi),cos(phi),0],[0,0,1]])
    T_2 = np.array([[1,0,X[0,0]],[0,1,X[1,0]],[0,0,1]])

    #Combine them
    T = T_2.dot(T_rot.dot(T_1))

    X_rot = T.dot(Xh)

    fig = plt.figure(figsize=(5,5))
    ax1 = fig.add_subplot(111)
    util.plot_object(ax1, X)
    util.plot_object(ax1, X_rot)
    ax1.set_xlim(ax1.get_ylim())
    ax1.grid()


# SECTION 2. Image transformation and least squares fitting

def image_transform_test():

    I = plt.imread('../data/cameraman.tif')

    # 45 deg. rotation around the image center
    T_1 = util.t2h(reg.identity(), 128*np.ones(2))
    T_2 = util.t2h(reg.rotate(np.pi/4), np.zeros(2))
    T_3 = util.t2h(reg.identity(), -128*np.ones(2))
    T_rot = T_1.dot(T_2).dot(T_3)

    # 45 deg. rotation around the image center followed by shearing
    T_shear = util.t2h(reg.shear(0.0, 0.5), np.zeros(2)).dot(T_rot)

    # scaling in the x direction and translation
    T_scale = util.t2h(reg.scale(1.5, 1), np.array([10,20]))

    It1, Xt1 = reg.image_transform(I, T_rot)
    It2, Xt2 = reg.image_transform(I, T_shear)
    It3, Xt3 = reg.image_transform(I, T_scale)

    fig = plt.figure(figsize=(12,5))

    ax1 = fig.add_subplot(131)
    im11 = ax1.imshow(I)
    im12 = ax1.imshow(It1, alpha=0.7)

    ax2 = fig.add_subplot(132)
    im21 = ax2.imshow(I)
    im22 = ax2.imshow(It2, alpha=0.7)

    ax3 = fig.add_subplot(133)
    im31 = ax3.imshow(I)
    im32 = ax3.imshow(It3, alpha=0.7)

    ax1.set_title('Rotation')
    ax2.set_title('Shearing')
    ax3.set_title('Scaling')

def ls_solve_test():

    #Define known variable matrix A
    A = np.array([[3,4],[5,6],[7,8],[17,10]])
    b = np.array([[1],[2],[3],[4]])

    w, E = reg.ls_solve(A,b)
    return(w)

def ls_affine_test():

    X = util.test_object(1)

    # convert to homogeneous coordinates
    Xh = util.c2h(X)

    T_rot = reg.rotate(np.pi/4)
    T_scale = reg.scale(1.2, 0.9)
    T_shear = reg.shear(0.2, 0.1)

    T = util.t2h(T_rot.dot(T_scale).dot(T_shear), [10, 20])

    Xm = T.dot(Xh)

    Te = reg.ls_affine(Xh, Xm)

    Xmt = Te.dot(Xm);

    fig = plt.figure(figsize=(12,5))

    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    util.plot_object(ax1, Xh)
    util.plot_object(ax2, Xm)
    util.plot_object(ax3, Xmt)

    ax1.set_title('Test shape')
    ax2.set_title('Arbitrary transformation')
    ax3.set_title('Retrieved test shape')

    ax1.grid()
    ax2.grid()
    ax3.grid()


# SECTION 3. Image similarity metrics

def correlation_test():

    I = plt.imread('../data/t1_demo.tif')
    Th = util.t2h(reg.identity(), np.array([10,20]))
    J, _ = reg.image_transform(I, Th)
    K, _ = reg.image_transform(J, Th)

    C1 = reg.correlation(I, I)
    C2 = reg.correlation(J,J)
    # the self correlation should be very close to 1
    assert abs(C1 - 1) < 10e-10, "Correlation function is incorrectly implemented (self correlation test)"
    assert abs(C2 - 1) < 10e-10, "Correlation function is incorrectly implemented (self correlation test)"

    C3 = reg.correlation(I,J)
    C4 = reg.correlation(I,K)
    print("Correlation between images with slight translocation:",C3)
    print("Correlation between images with big translocation:", C4, "\n")

    print('Test successful!')

    ########
    print("\n\nAnd now for a test on the joint probability histogram: \n")
    p = reg.joint_histogram(I,J)
    assert abs(1-float(np.sum(p))) < 0.01, "Mass probability function error..."
    print("Test successful!")


def mutual_information_test():

    I = plt.imread('../data/cameraman.tif')

    # mutual information of an image with itself
    p1 = reg.joint_histogram(I, I)
    MI1 = reg.mutual_information(p1)

    J = np.random.randint(255, size=(512, 512))
    K = np.random.randint(255, size=(512, 512))
    p2 = reg.joint_histogram(J, K)
    MI2 = reg.mutual_information(p2)


    print(MI1," ",MI2)
    assert MI1 > 0, "Mutual Information calculation error (MI<0).."
    assert MI2 > 0, "Mutual Information calculation error (MI<0).."

    print('Test successful!')


def mutual_information_e_test():

    I = plt.imread('../data/cameraman.tif')

    N1 = np.random.randint(255, size=(512, 512))
    N2 = np.random.randint(255, size=(512, 512))

    # mutual information of an image with itself
    p1 = reg.joint_histogram(I, I)
    MI1 = reg.mutual_information_e(p1)
    MI2 = reg.mutual_information(p1)
    assert abs(MI1-MI2) < 10e-3, "Mutual information function with entropy is incorrectly implemented (difference with reference implementation test)"

    print('Test successful!')


# SECTION 4. Towards intensity-based image registration

def test_fun(x,y):
    return (1-(x**2+y**3))*exp(-(x**2+y**2)/2)

def ngradient_test():

    # NOTE: test function not strictly scalar-valued
    exponential = lambda x: np.exp(x)
    g1 = reg.ngradient(exponential, np.ones((1,)))
    assert abs(g1 - exponential(1)) < 1e-5, "Numerical gradient is incorrectly implemented (exponential test)"

    g2 = reg.ngradient(test_fun,[1,2])

    print('Test successful!')


def registration_metrics_demo(use_t2=False):

    # read a T1 image
    I = plt.imread('../data/t1_demo.tif')

    if use_t2:
        # read the corresponding T2 image
        # note that the T1 and T2 images are already registered
        I_t2 = plt.imread('../data/t2_demo.tif')

    # create a linear space of rotation angles - 101 angles between 0 and 360 deg.
    angles = np.linspace(-np.pi, np.pi, 101, endpoint=True)

    CC = np.full(angles.shape, np.nan)
    MI = np.full(angles.shape, np.nan)

    # visualization
    fig = plt.figure(figsize=(14,6))

    # correlation
    ax1 = fig.add_subplot(131, xlim=(-np.pi, np.pi), ylim=(-1.1, 1.1))
    line1, = ax1.plot(angles, CC, lw=2)
    ax1.set_xlabel('Rotation angle')
    ax1.set_ylabel('Correlation coefficient')
    ax1.grid()

    # mutual mutual_information
    ax2  = fig.add_subplot(132, xlim=(-np.pi, np.pi), ylim=(0, 2))
    line2, = ax2.plot(angles, MI, lw=2)
    ax2.set_xlabel('Rotation angle')
    ax2.set_ylabel('Mutual information')
    ax2.grid()

    # images
    ax3 = fig.add_subplot(133)
    im1 = ax3.imshow(I)
    im2 = ax3.imshow(I, alpha=0.7)

    # used for rotation around image center
    t = np.array([I.shape[0], I.shape[1]])/2 + 0.5
    T_1 = util.t2h(reg.identity(), t)
    T_3 = util.t2h(reg.identity(), -t)

    # loop over the rotation angles
    for k, ang in enumerate(angles):
        # transformation matrix for rotating the image
        # I by angles(k) around its center point
        T_2 = util.t2h(reg.rotate(ang), np.zeros(2))
        T_rot = T_1.dot(T_2).dot(T_3)

        if use_t2:
            # rotate the T2 image
            J, Xt = reg.image_transform(I_t2, T_rot)
        else:
            # rotate the T1 image
            J, Xt = reg.image_transform(I, T_rot)

        # compute the joint histogram with 16 bins
        p = reg.joint_histogram(I, J, 16, [0, 255])

        CC[k] = reg.correlation(I, J)
        MI[k] = reg.mutual_information(p)

        clear_output(wait = True)

        # visualize the results
        line1.set_ydata(CC)
        line2.set_ydata(MI)
        im2.set_data(J)

        display(fig)
