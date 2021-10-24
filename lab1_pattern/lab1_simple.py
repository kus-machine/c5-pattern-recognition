from numba import njit, prange
import numpy as np
from cv2 import imread, imwrite, imshow, waitKey, COLOR_BGR2RGB, cvtColor, vconcat, hconcat, putText,\
    FONT_HERSHEY_SIMPLEX, resize
#import cv2
import matplotlib.pyplot as plt
import random
import time
from os import mkdir, rmdir, path

#func that show the image (put in BGR (cv2))
# def show_image(image,size=(9,7)):
#     plt.figure(figsize=size)
#     #Before showing image, bgr color order transformed to rgb order
#     plt.imshow(cvtColor(image, COLOR_BGR2RGB))
#     plt.xticks([])
#     plt.yticks([])
#     plt.show()



def take_params(image, percent_h = .2, percent_w = .5, take_mode = "corner_square"):
    if take_mode == "corner_square":
        piece0=image[:int(percent_h*image.shape[0]),:int(percent_w*image.shape[1])]
        piece1=image[int(-percent_h*image.shape[0]):,int(-percent_w*image.shape[1]):]
    elif take_mode == "full_line":
        piece0=image[:int(percent_h*image.shape[0]),:]
        piece1=image[int(-percent_h*image.shape[0]):,:]

    piece0=np.reshape(piece0,(piece0.shape[0]*piece0.shape[1],piece0.shape[2]))
    piece1=np.reshape(piece1,(piece1.shape[0]*piece1.shape[1],piece1.shape[2]))

    mean0=np.mean(piece0, axis=0)
    mean1=np.mean(piece1, axis=0)

    cov0=np.cov([piece0[...,0],piece0[...,1],piece0[...,2]])
    cov1=np.cov([piece1[...,0],piece1[...,1],piece1[...,2]])
        
    return mean0, mean1, cov0, cov1

@njit(cache=True, nogil=False, fastmath=True, parallel=True)
def pdf_multivariate_gauss(x, mu, cov): 
    # assert(mu.shape[0] > mu.shape[1]), 'mu must be a row vector'
    # assert(x.shape[0] > x.shape[1]), 'x must be a row vector'
    # assert(cov.shape[0] == cov.shape[1]), 'covariance matrix must be square'
    # assert(mu.shape[0] == cov.shape[0]), 'cov_mat and mu_vec must have the same dimensions'
    # assert(mu.shape[0] == x.shape[0]), 'mu and x must have the same dimensions'
    part1 = 1 / ( ((2* np.pi)**(mu.shape[0]/2)) * (np.linalg.det(cov)**(1/2)) )
    part2 = (-1/2) * (((x-mu).T).dot(np.linalg.inv(cov))).dot((x-mu))
    return np.double(part1 * np.exp(part2))

@njit(cache=True, nogil=False, fastmath=True, parallel=False)
def g_multiply(LABELS, eps, i, j, label):
    left = right = top = bot = 1
    if (i > 0):
        if(LABELS[i+1][j] == label):
            bot = eps
        else:
            bot = 1 - eps
    if (i < LABELS.shape[0] - 1):
        if(LABELS[i-1][j] == label):
            top = eps
        else:
            top = 1 - eps
    if (j > 0):
        if(LABELS[i][j-1] == label):
            left = eps
        else:
            left = 1 - eps
    if (j < LABELS.shape[1] - 1):
        if(LABELS[i][j+1] == label):
            right = eps
        else:
            right = 1 - eps

    return left * right * top * bot

@njit(cache=True, nogil=False, fastmath=True, parallel=True)
def sampler_1st_iter(image, mean0, mean1, cov0, cov1):
    image_resh=image.reshape(image.shape[0]*image.shape[1], image.shape[2])
    LABELS=np.zeros(image_resh.shape[0], np.uint8)
    PROB = np.zeros((image_resh.shape[0],2), np.double)
    for i in prange(image_resh.shape[0]-1):
        #for j in prange(image.shape[1]-1):
        PROB[i][0] = pdf_multivariate_gauss(image_resh[i], mean0, cov0)
        PROB[i][1] = pdf_multivariate_gauss(image_resh[i], mean1, cov1)
        c = np.random.uniform(0, PROB[i][0] + PROB[i][1])
        if c < PROB[i][0]:
            LABELS[i] = 1
    return LABELS.reshape(image.shape[:2]), PROB.reshape((image.shape[0],image.shape[1], 2))


@njit(cache=True, nogil=False, fastmath=True, parallel=True)
def sampler_one_iter(PROB, LABELS, eps):
    new_LABELS = np.zeros_like(LABELS)
    for i in prange(PROB.shape[0]):
        for j in prange(PROB.shape[1]):
            a=PROB[i][j][0]*g_multiply(LABELS, eps, i, j, 0)
            b=PROB[i][j][1]*g_multiply(LABELS, eps, i, j, 1)
            c = np.random.uniform(0,a+b)
            if c < a:
                new_LABELS[i][j] = 1
    return new_LABELS


def sampler(image, eps=0.2, scale=[256, 256], n_iter=100, percent_h = .2, percent_w = .5, take_mode = "corner_square"):

    #scaling image to lower resolution for good performance
    if(scale!=[0,0]):
        img_for_sampl = resize(image, (scale[0], scale[1]))
    else:
        img_for_sampl = image

    #creating a mas of all labels per iterations
    mas = np.zeros((n_iter, img_for_sampl.shape[0], img_for_sampl.shape[1]))

    #take params from pieces (from sky and filed of grass)
    mean0, mean1, cov0, cov1 = take_params(image, percent_h = percent_h, percent_w = percent_w, take_mode = take_mode)

    #create a matrix of probabilities of each class
    LABELS_0, PROB_0 = sampler_1st_iter(img_for_sampl, mean0, mean1, cov0, cov1)
    mas[0] = LABELS_0

    #sampling
    for i in range(n_iter-2):
        mas[i+1] = sampler_one_iter(PROB_0, mas[i], eps)

    return img_for_sampl, mas

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
filename = "field6.jpg"
img1 = imread(filename)
scale = [512, 512]
n_iter = 100
eps = 0.1
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

start_time = time.time()


img_for_sampl, LABELS = sampler(img1, eps = eps, scale = scale, n_iter = n_iter, percent_h = .4, take_mode = "full_line")


alg_time = time.time() - start_time
print("--- %s seconds ---" % alg_time)
print("time per iter: ", alg_time/n_iter)

if (not path.exists(filename[:-4] + '/')):
    mkdir(filename[:-4] + '/')
    print("Directory " + filename[:-4] + '/' "created.")
result=np.zeros_like(img_for_sampl)
for i in range(LABELS.shape[0]-1):
    result[...,0] = result[...,1] = LABELS[i] * 255
    result[result[...,0] == 0]=[0, 0, 255]
    result2 = hconcat([(0.5 * img_for_sampl + result * 0.5).astype(np.uint8), img_for_sampl])
    result2 = putText(result2,
        str(i),
        (int(img_for_sampl.shape[0]*0.05), int(img_for_sampl.shape[1]*0.1)),
        FONT_HERSHEY_SIMPLEX,
        1.5,
        (0, 255, 255),
        4)
    imwrite(filename[:-4] + '/' + str(i) + '.png', result2)

