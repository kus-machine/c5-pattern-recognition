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
def show_image(image,size=(9,7)):
    plt.figure(figsize=size)
    #Before showing image, bgr color order transformed to rgb order
    plt.imshow(cvtColor(image, COLOR_BGR2RGB))
    plt.xticks([])
    plt.yticks([])
    plt.show()

#take 2 pieces from top and bot of img 
@njit(cache=True, nogil=True, fastmath=True)
def take_piece(image, percent_h = .1, percent_w = .5, take_mode = "corner_square"):
    if take_mode == "corner_square":
        upper=image[:int(percent_h*image.shape[0]),:int(percent_w*image.shape[1])]
        lower=image[int(-percent_h*image.shape[0]):,int(-percent_w*image.shape[1]):]
    elif take_mode == "full_line":
        upper=image[:int(percent_h*image.shape[0]),:]
        lower=image[int(-percent_h*image.shape[0]):,:]
        
    return upper,lower

# #@njit(cache=True, nogil=True, fastmath=True)
# def norm_pdf_multivariate(x, mu, sigma):
#     size = len(x)
#     if size == len(mu) and (size, size) == sigma.shape:
#         det = np.linalg.det(sigma)
#         if det == 0:
#             raise NameError("The covariance matrix can't be singular")

#         norm_const = 1.0/ ( np.math.pow((2*np.pi),float(size)/2) * np.math.pow(det,1.0/2) )
#         x_mu = np.matrix(x - mu)
#         inv = np.linalg.inv(sigma)
#         result = np.math.pow(np.math.e, -0.5 * (x_mu * inv * x_mu.T))
#         return norm_const * result
#     else:
#         raise NameError("The dimensions of the input don't match")


@njit(cache=True, nogil=False, fastmath=True)
def pdf_multivariate_gauss(x, mu, cov): 
    # assert(mu.shape[0] > mu.shape[1]), 'mu must be a row vector'
    # assert(x.shape[0] > x.shape[1]), 'x must be a row vector'
    # assert(cov.shape[0] == cov.shape[1]), 'covariance matrix must be square'
    # assert(mu.shape[0] == cov.shape[0]), 'cov_mat and mu_vec must have the same dimensions'
    # assert(mu.shape[0] == x.shape[0]), 'mu and x must have the same dimensions'
    part1 = 1 / ( ((2* np.pi)**(mu.shape[0]/2)) * (np.linalg.det(cov)**(1/2)) )
    part2 = (-1/2) * (((x-mu).T).dot(np.linalg.inv(cov))).dot((x-mu))
    return float(part1 * np.exp(part2))


@njit(cache=True, nogil=False, fastmath=True, parallel=False)
def sampler_1st_iter(image, mean1, mean2, cov1, cov2):
    image_resh=image.reshape(image.shape[0]*image.shape[1], image.shape[2])
    LABELS=np.zeros(image_resh.shape[0], np.uint8)
    for i in prange(image_resh.shape[0]-1):
        #for j in prange(image.shape[1]-1):
        a = pdf_multivariate_gauss(image_resh[i], mean1, cov1)
        b = pdf_multivariate_gauss(image_resh[i], mean2, cov2)
        #c = np.random.uniform(0,a+b)
        if a>b:
            LABELS[i] = 1
    return LABELS.reshape(image.shape[:2])



@njit(cache=True, nogil=False, fastmath=True, parallel=True)
def sampler_one_iter(image, mean1, mean2, cov1, cov2):
    LABELS=np.zeros(image.shape[:2], np.uint8)
    for i in prange(image.shape[0]-1):
        for j in range(image.shape[1]-1):
            if(i==0):
                if(j==0):
                    a = pdf_multivariate_gauss(image[i+1][j], mean1, cov1)*pdf_multivariate_gauss(image[i][j+1], mean1, cov1)
                    b = pdf_multivariate_gauss(image[i+1][j], mean2, cov2)*pdf_multivariate_gauss(image[i][j+1], mean2, cov2)
                    c = np.random.uniform(0,a+b)
                    if c < a:
                        LABELS[i][j] = 1
                    # else:
                    #     LABELS[i][j] = 0
                elif(j==image.shape[1]-1):
                    a = pdf_multivariate_gauss(image[i+1][j], mean1, cov1)*pdf_multivariate_gauss(image[i][j-1], mean1, cov1)
                    b = pdf_multivariate_gauss(image[i+1][j], mean2, cov2)*pdf_multivariate_gauss(image[i][j-1], mean2, cov2)
                    c = np.random.uniform(0,a+b)
                    if c < a:
                        LABELS[i][j] = 1
                    # else:
                    #     LABELS[i][j] = 0
                else:
                    a = pdf_multivariate_gauss(image[i+1][j], mean1, cov1)*pdf_multivariate_gauss(image[i][j+1], mean1, cov1)*pdf_multivariate_gauss(image[i][j-1], mean1, cov1)
                    b = pdf_multivariate_gauss(image[i+1][j], mean2, cov2)*pdf_multivariate_gauss(image[i][j+1], mean2, cov2)*pdf_multivariate_gauss(image[i][j-1], mean2, cov2)
                    c = np.random.uniform(0,a+b)
                    if c < a:
                        LABELS[i][j] = 1
                    # else:
                    #     LABELS[i][j] = 0
            elif(i==image.shape[0]-1):
                if(j==0):
                    a = pdf_multivariate_gauss(image[i-1][j], mean1, cov1)*pdf_multivariate_gauss(image[i][j+1], mean1, cov1)
                    b = pdf_multivariate_gauss(image[i-1][j], mean2, cov2)*pdf_multivariate_gauss(image[i][j+1], mean2, cov2)
                    c = np.random.uniform(0,a+b)
                    if c < a:
                        LABELS[i][j] = 1
                    # else:
                    #     LABELS[i][j] = 0
                elif(j==image.shape[1]-1):
                    a = pdf_multivariate_gauss(image[i-1][j], mean1, cov1)*pdf_multivariate_gauss(image[i][j-1], mean1, cov1)
                    b = pdf_multivariate_gauss(image[i-1][j], mean2, cov2)*pdf_multivariate_gauss(image[i][j-1], mean2, cov2)
                    c = np.random.uniform(0,a+b)
                    if c < a:
                        LABELS[i][j] = 1
                    # else:
                    #     LABELS[i][j] = 0
                else:
                    a = pdf_multivariate_gauss(image[i-1][j], mean1, cov1)*pdf_multivariate_gauss(image[i][j+1], mean1, cov1)*pdf_multivariate_gauss(image[i][j-1], mean1, cov1)
                    b = pdf_multivariate_gauss(image[i-1][j], mean2, cov2)*pdf_multivariate_gauss(image[i][j+1], mean2, cov2)*pdf_multivariate_gauss(image[i][j-1], mean2, cov2)
                    c = np.random.uniform(0,a+b)
                    if c < a:
                        LABELS[i][j] = 1
                    # else:
                    #     LABELS[i][j] = 0

            else:
                if(j==0):
                    a = pdf_multivariate_gauss(image[i+1][j], mean1, cov1)*pdf_multivariate_gauss(image[i-1][j], mean1, cov1)*pdf_multivariate_gauss(image[i][j+1], mean1, cov1)
                    b = pdf_multivariate_gauss(image[i+1][j], mean2, cov2)*pdf_multivariate_gauss(image[i-1][j], mean2, cov2)*pdf_multivariate_gauss(image[i][j+1], mean2, cov2)
                    c = np.random.uniform(0,a+b)
                    if c < a:
                        LABELS[i][j] = 1
                    # else:
                    #     LABELS[i][j] = 0
                elif(j==image.shape[1]-1):
                    a = pdf_multivariate_gauss(image[i+1][j], mean1, cov1)*pdf_multivariate_gauss(image[i-1][j], mean1, cov1)*pdf_multivariate_gauss(image[i][j-1], mean1, cov1)
                    b = pdf_multivariate_gauss(image[i+1][j], mean2, cov2)*pdf_multivariate_gauss(image[i-1][j], mean2, cov2)*pdf_multivariate_gauss(image[i][j-1], mean2, cov2)
                    c = np.random.uniform(0,a+b)
                    if c < a:
                        LABELS[i][j] = 1
                    # else:
                    #     LABELS[i][j] = 0
                else:
                    a = pdf_multivariate_gauss(image[i+1][j], mean1, cov1)*pdf_multivariate_gauss(image[i-1][j], mean1, cov1)*pdf_multivariate_gauss(image[i][j+1], mean1, cov1)*pdf_multivariate_gauss(image[i][j-1], mean1, cov1)
                    b = pdf_multivariate_gauss(image[i+1][j], mean2, cov2)*pdf_multivariate_gauss(image[i-1][j], mean2, cov2)*pdf_multivariate_gauss(image[i][j+1], mean2, cov2)*pdf_multivariate_gauss(image[i][j-1], mean2, cov2)
                    c = np.random.uniform(0,a+b)
                    if c < a:
                        LABELS[i][j] = 1
                    # else:
                    #     LABELS[i][j] = 0
    return LABELS

#@njit
def update_params(image, labels, printing=False):
    class1=image[labels==1]
    class2=image[labels==0]
    mean1=np.mean(class1, axis=0)
    mean2=np.mean(class2, axis=0)    
    cov1=np.cov([class1[...,0],class1[...,1],class1[...,2]])
    cov2=np.cov([class2[...,0],class2[...,1],class2[...,2]])
    if(printing==True):
        print("means 1st class: ", mean1)
    return mean1, mean2, cov1, cov2

#@njit(cache=True, nogil=True, fastmath=True, parallel=False)
def G_sampler(image, m1, m2, c1, c2, n_iter):
    mas=np.zeros((n_iter,image.shape[0], image.shape[1], image.shape[2]))
    #LABELS=np.zeros(image.shape[:2])
    LABELS = sampler_1st_iter(image, m1, m2, c1, c2)
    for k in range(n_iter):
        print(k, " iteration:")
        new_mean1, new_mean2, new_cov1, new_cov2 = update_params(image, LABELS)
        LABELS = sampler_one_iter(image, new_mean1, new_mean2, new_cov1, new_cov2)
        #print("labels shape: ", LABELS.shape)
        result[...,0] = result[...,1] = LABELS[::]*255
        result[result[...,0]==0]=[0,0,255]
        
        mas[k]=result
        
    return mas

filename = "field6.jpg"
img1 = imread(filename)
scale_h, scale_w = 256, 256
n_iter = 30
img_for_sampl = resize(img1, (scale_h, scale_w))

start_time = time.time()


result=np.zeros_like(img_for_sampl)
#first taking of parametrs from picture's pieces:
piece1, piece2 = take_piece(img_for_sampl, take_mode = "corner_square")
mean1=np.mean(piece1, axis=(0,1))
mean2=np.mean(piece2, axis=(0,1))

im_sky_for_cov=np.reshape(piece1,(piece1.shape[0]*piece1.shape[1],piece1.shape[2]))
im_gr_for_cov=np.reshape(piece2,(piece2.shape[0]*piece2.shape[1],piece2.shape[2]))
cov_sky=np.cov([im_sky_for_cov[...,0],im_sky_for_cov[...,1],im_sky_for_cov[...,2]])
cov_gr=np.cov([im_gr_for_cov[...,0],im_gr_for_cov[...,1],im_gr_for_cov[...,2]])

sampl=G_sampler(img_for_sampl, mean1, mean2, cov_sky, cov_gr, n_iter=n_iter)

alg_time = time.time() - start_time
print("--- %s seconds ---" % alg_time)
print("time per iter: ", alg_time/n_iter)

if (not path.exists(filename[:-4] + '_wrong/')):
    mkdir(filename[:-4] + '_wrong/')
for i in range(sampl.shape[0]):
    #print(sampl.shape, sampl.sum())
    result2 = hconcat([(0.5*img_for_sampl + sampl[i]*0.5).astype(np.uint8), img_for_sampl])
    result2 = putText(result2,
        str(i),
        (int(img_for_sampl.shape[0]*0.05), int(img_for_sampl.shape[1]*0.1)),
        FONT_HERSHEY_SIMPLEX,
        1.5,
        (0, 255, 255),
        4)
    imwrite(filename[:-4] + '_wrong/' + str(i) + '.png', result2)