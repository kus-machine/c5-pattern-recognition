from numba import njit, prange
import numpy as np
from cv2 import imread, imwrite, imshow, waitKey, COLOR_BGR2RGB, cvtColor, vconcat, hconcat, putText, \
    FONT_HERSHEY_SIMPLEX, resize
# import cv2
import matplotlib.pyplot as plt
import time
# from os import mkdir, rmdir, path


# func that show the image (put in BGR (cv2))
def show_image(image, size=(9, 7)):
    plt.figure(figsize=size)
    # Before showing image, bgr color order transformed to rgb order
    plt.imshow(cvtColor(image, COLOR_BGR2RGB))
    plt.xticks([])
    plt.yticks([])
    plt.show()


# take parameters of multi-gauss distribution of 2 classes (sky and ground) for good axis
def take_params(image, percent_h=.2, percent_w=.5, take_mode="corner_square"):
    if take_mode == "corner_square":
        piece0 = image[:int(percent_h * image.shape[0]), :int(percent_w * image.shape[1])]
        piece1 = image[int(-percent_h * image.shape[0]):, int(-percent_w * image.shape[1]):]
    elif take_mode == "full_line":
        piece0 = image[:int(percent_h * image.shape[0]), :]
        piece1 = image[int(-percent_h * image.shape[0]):, :]

    piece0 = np.reshape(piece0, (piece0.shape[0] * piece0.shape[1], piece0.shape[2]))
    piece1 = np.reshape(piece1, (piece1.shape[0] * piece1.shape[1], piece1.shape[2]))

    mean_0 = np.mean(piece0, axis=0)
    mean_1 = np.mean(piece1, axis=0)

    cov_0 = np.cov([piece0[..., 0], piece0[..., 1], piece0[..., 2]])
    cov_1 = np.cov([piece1[..., 0], piece1[..., 1], piece1[..., 2]])

    return mean_0, mean_1, cov_0, cov_1


# return logarithm of multi-gauss distribution (simplified on sheet of paper)
@njit(fastmath=True, cache=False)
def pdf_multivariate_gauss_log(x, mu, cov):
    # part1 = 1 / (((2 * np.pi) ** (mu.shape[0] / 2)) * (np.linalg.det(cov) ** (1 / 2)))
    # part2 = (-1 / 2) * (((x - mu).T).dot(np.linalg.inv(cov))).dot((x - mu))
    # return np.double(part1 * np.exp(part2))
    # _____________________________________________________________________
    # take log() from gauss distr: log(exp(blablabla)/Z) = blablabla - log(Z)
    part1 = np.sqrt(np.linalg.det(cov))
    part2 = (-1 / 2) * ((x - mu).T.dot(np.linalg.inv(cov))).dot((x - mu))
    return part2 - part1


# @njit(cache=False, nogil=False, fastmath=True, parallel=True)
@njit(cache=False, fastmath=True)
def get_probs(image, mean_0, mean_1, cov_0, cov_1):
    image_resh = image.reshape(image.shape[0] * image.shape[1], image.shape[2])
    PROB = np.zeros((image_resh.shape[0], 2), np.double)
    for i in range(image_resh.shape[0]):
        # for j in prange(image.shape[1]-1):
        PROB[i, 0] = pdf_multivariate_gauss_log(image_resh[i], mean_0, cov_0)
        PROB[i, 1] = pdf_multivariate_gauss_log(image_resh[i], mean_1, cov_1)
    return PROB.reshape((image.shape[0], image.shape[1], 2))


# this func only for 1st iteration (fill start graph)
def n_of_Nt(i, j, shape):
    n = 4
    if i == 0:
        n -= 1
    if i == (shape[0] - 1):
        n -= 1
    if j == 0:
        n -= 1
    if j == (shape[1] - 1):
        n -= 1
    return n


def fill_graph_edges(probs, epsilon=0.25):
    g_for_im = np.zeros((2 * probs.shape[0] - 1, 2 * probs.shape[1] - 1, 4), np.float32)
    # 4 edges - 0-0, 0-1, 1-0, 1-1 in this order
    for i in range(g_for_im.shape[0]):
        # четные i; смотрим объекты слева и справа
        if i % 2 == 0:
            for j in range(1, g_for_im.shape[1] - 1, 2):
                g_for_im[i, j, 0] = (probs[i // 2, (j - 1) // 2, 0] + probs[i // 2, (j + 1) // 2, 0]) / n_of_Nt(
                    i, j, g_for_im.shape) + np.log(epsilon)
                g_for_im[i, j, 1] = (probs[i // 2, (j - 1) // 2, 0] + probs[i // 2, (j + 1) // 2, 1]) / n_of_Nt(
                    i, j, g_for_im.shape) + np.log(1 - epsilon)
                g_for_im[i, j, 2] = (probs[i // 2, (j - 1) // 2, 1] + probs[i // 2, (j + 1) // 2, 0]) / n_of_Nt(
                    i, j, g_for_im.shape) + np.log(1 - epsilon)
                g_for_im[i, j, 3] = (probs[i // 2, (j - 1) // 2, 1] + probs[i // 2, (j + 1) // 2, 1]) / n_of_Nt(
                    i, j, g_for_im.shape) + np.log(epsilon)
        # нечетные i; смотрим объекты сверху и снизу
        if i % 2 != 0:
            for j in range(0, g_for_im.shape[1], 2):
                g_for_im[i, j, 0] = (probs[(i - 1) // 2, j // 2, 0] + probs[(i + 1) // 2, j // 2, 0]) / n_of_Nt(
                    i, j, g_for_im.shape) + np.log(epsilon)
                g_for_im[i, j, 1] = (probs[(i - 1) // 2, j // 2, 0] + probs[(i + 1) // 2, j // 2, 1]) / n_of_Nt(
                    i, j, g_for_im.shape) + np.log(1 - epsilon)
                g_for_im[i, j, 2] = (probs[(i - 1) // 2, j // 2, 1] + probs[(i + 1) // 2, j // 2, 0]) / n_of_Nt(
                    i, j, g_for_im.shape) + np.log(1 - epsilon)
                g_for_im[i, j, 3] = (probs[(i - 1) // 2, j // 2, 1] + probs[(i + 1) // 2, j // 2, 1]) / n_of_Nt(
                    i, j, g_for_im.shape) + np.log(epsilon)
    return g_for_im


@njit(fastmath=True)
def big_sum(i, j, n_ed, g):
    summa = 0
    n_nei = 8
    # calculate number of neirb for t and t' both:
    if i == 0 or i == g.shape[0] - 1:
        n_nei -= 2
        if j == 1 or j == g.shape[1] - 2:
            n_nei -= 1
    if j == 0 or j == g.shape[1] - 1:
        n_nei -= 2
        if i == 1 or i == g.shape[0] - 2:
            n_nei -= 1

    # firstly, calculate first side of edge g(i,j): from t to t' (we think that current object is t)
    # start t -> t'

    # case: edges from t to t': 0-0 or 0-1 - looking for edges from t to t'' 0-0 or 0-1
    if n_ed == 0 or n_ed == 1:
        summa += max(g[i, j, 0], g[i, j, 1])
        # horiz edges - watch for t - left
        if i % 2 == 0:
            if i == 0:
                if j % 2 == 1:
                    if j == 1:
                        summa += max(g[i + 1, j - 1, 0], g[i + 1, j - 1, 1])
                    else:
                        summa += max(g[i + 1, j - 1, 0], g[i + 1, j - 1, 1]) + max(g[i, j - 2, 0], g[i, j - 2, 1])
            elif i == g.shape[0] - 1:
                if j % 2 == 1:
                    if j == 1:
                        summa += max(g[i - 1, j - 1, 0], g[i - 1, j - 1, 1])
                    else:
                        summa += max(g[i - 1, j - 1, 0], g[i - 1, j - 1, 1]) + max(g[i, j - 2, 0], g[i, j - 2, 1])
            else:
                if j % 2 == 1:
                    if j == 1:
                        summa += max(g[i - 1, j - 1, 0], g[i - 1, j - 1, 1]) + max(g[i + 1, j - 1, 0],
                                                                                   g[i + 1, j - 1, 1])
                    else:
                        summa += max(g[i - 1, j - 1, 0], g[i - 1, j - 1, 1]) + max(g[i + 1, j - 1, 0],
                                                                                   g[i + 1, j - 1, 1]) + max(
                            g[i, j - 2, 0], g[i, j - 2, 1])
        # vert edges - watch for t - top
        elif i % 2 == 1:
            if i == 1:
                if j % 2 == 0:
                    if j == 0:
                        summa += max(g[i - 1, j + 1, 0], g[i - 1, j + 1, 1])
                    elif j == g.shape[1] - 1:
                        summa += max(g[i - 1, j - 1, 0], g[i - 1, j - 1, 1])
                    else:
                        summa += max(g[i - 1, j - 1, 0], g[i - 1, j - 1, 1]) + max(g[i - 1, j + 1, 0],
                                                                                   g[i - 1, j + 1, 1])
            else:
                if j % 2 == 0:
                    if j == 0:
                        summa += max(g[i - 1, j + 1, 0], g[i - 1, j + 1, 1]) + max(g[i - 2, j, 0], g[i - 2, j, 1])
                    elif j == g.shape[1] - 1:
                        summa += max(g[i - 1, j - 1, 0], g[i - 1, j - 1, 1]) + max(g[i - 2, j, 0], g[i - 2, j, 1])
                    else:
                        summa += max(g[i - 1, j + 1, 0], g[i - 1, j + 1, 1]) + max(g[i - 1, j - 1, 0],
                                                                                   g[i - 1, j - 1, 1]) + max(
                                                                                   g[i - 2, j, 0], g[i - 2, j, 1])
    # _________________________________________________________________________________________________________________
    # case: edges from t to t': 1-0 or 1-1 - looking for edges from t to t'' 1-0 or 1-1
    elif n_ed == 2 or n_ed == 3:
        summa += max(g[i, j, 2], g[i, j, 3])
        # horiz edges - watch for t - left
        if i % 2 == 0:
            if i == 0:
                if j % 2 == 1:
                    if j == 1:
                        summa += max(g[i + 1, j - 1, 2], g[i + 1, j - 1, 3])
                    else:
                        summa += max(g[i + 1, j - 1, 2], g[i + 1, j - 1, 3]) + max(g[i, j - 2, 2], g[i, j - 2, 3])
            elif i == g.shape[0] - 1:
                if j % 2 == 1:
                    if j == 1:
                        summa += max(g[i - 1, j - 1, 2], g[i - 1, j - 1, 3])
                    else:
                        summa += max(g[i - 1, j - 1, 2], g[i - 1, j - 1, 3]) + max(g[i, j - 2, 2], g[i, j - 2, 3])
            else:
                if j % 2 == 1:
                    if j == 1:
                        summa += max(g[i - 1, j - 1, 2], g[i - 1, j - 1, 3]) + max(g[i + 1, j - 1, 2],
                                                                                   g[i + 1, j - 1, 3])
                    else:
                        summa += max(g[i - 1, j - 1, 2], g[i - 1, j - 1, 3]) + max(g[i + 1, j - 1, 2],
                                                                                   g[i + 1, j - 1, 3]) + max(
                                                                                   g[i, j - 2, 2],
                                                                                   g[i, j - 2, 3])
        # vert edges - watch for t - top
        elif i % 2 == 1:
            if i == 1:
                if j % 2 == 0:
                    if j == 0:
                        summa += max(g[i - 1, j + 1, 2], g[i - 1, j + 1, 3])
                    elif j == g.shape[1] - 1:
                        summa += max(g[i - 1, j - 1, 2], g[i - 1, j - 1, 3])
                    else:
                        summa += max(g[i - 1, j - 1, 2], g[i - 1, j - 1, 3]) + max(g[i - 1, j + 1, 2],
                                                                                   g[i - 1, j + 1, 3])
            else:
                if j % 2 == 0:
                    if j == 0:
                        summa += max(g[i - 1, j + 1, 2], g[i - 1, j + 1, 3]) + max(g[i - 2, j, 2], g[i - 2, j, 3])
                    elif j == g.shape[1] - 1:
                        summa += max(g[i - 1, j - 1, 2], g[i - 1, j - 1, 3]) + max(g[i - 2, j, 2], g[i - 2, j, 3])
                    else:
                        summa += max(g[i - 1, j + 1, 2], g[i - 1, j + 1, 3]) + max(g[i - 1, j - 1, 2],
                                                                                   g[i - 1, j - 1, 3]) + max(
                                                                                   g[i - 2, j, 2],
                                                                                   g[i - 2, j, 3])
    # end t -> t'
    # -----------------------------------------------------------------------------------------------------------------
    # start t' -> t
    # now, calculate second side of edge g(i,j): from t' to t (we think that current object is t')

    # case: edges from t' to t: 0-0 or 0-1 (same as t->t': 0-0 or 1-0) - looking for edges from t' to t'' 0-0 or 0-1
    if n_ed == 0 or n_ed == 2:
        summa += max(g[i, j, 0], g[i, j, 2])
        # horiz edges
        if i % 2 == 0:
            if i == 0:
                if j % 2 == 1:
                    if j == g.shape[1] - 2:
                        summa += max(g[i + 1, j + 1, 0], g[i + 1, j + 1, 1])
                    else:
                        summa += max(g[i + 1, j + 1, 0], g[i + 1, j + 1, 1]) + max(g[i, j + 2, 0], g[i, j + 2, 1])
            elif i == g.shape[0] - 1:
                if j % 2 == 1:
                    if j == g.shape[1] - 2:
                        summa += max(g[i - 1, j + 1, 0], g[i - 1, j + 1, 1])
                    else:
                        summa += max(g[i - 1, j + 1, 0], g[i - 1, j + 1, 1]) + max(g[i, j + 2, 0], g[i, j + 2, 1])
            else:
                if j % 2 == 1:
                    if j == g.shape[1] - 2:
                        summa += max(g[i - 1, j + 1, 0], g[i - 1, j + 1, 1]) + max(g[i + 1, j + 1, 0],
                                                                                   g[i + 1, j + 1, 1])
                    else:
                        summa += max(g[i - 1, j + 1, 0], g[i - 1, j + 1, 1]) + max(g[i + 1, j + 1, 0],
                                                                                   g[i + 1, j + 1, 1]) + max(
                                                                                   g[i, j + 2, 0], g[i, j + 2, 1])
        # vert edges
        elif i % 2 == 1:
            if i == g.shape[0] - 2:
                if j % 2 == 0:
                    if j == 0:
                        summa += max(g[i + 1, j + 1, 0], g[i + 1, j + 1, 1])
                    elif j == g.shape[1] - 1:
                        summa += max(g[i + 1, j - 1, 0], g[i + 1, j - 1, 1])
                    else:
                        summa += max(g[i + 1, j + 1, 0], g[i + 1, j + 1, 1]) + max(g[i + 1, j - 1, 0],
                                                                                   g[i + 1, j - 1, 1])
            else:
                if j % 2 == 0:
                    if j == 0:
                        summa += max(g[i + 1, j + 1, 0], g[i + 1, j + 1, 1]) + max(g[i + 2, j, 0], g[i + 2, j, 1])
                    elif j == g.shape[1] - 1:
                        summa += max(g[i + 1, j - 1, 0], g[i + 1, j - 1, 1]) + max(g[i + 2, j, 0], g[i + 2, j, 1])
                    else:
                        summa += max(g[i + 1, j + 1, 0], g[i + 1, j + 1, 1]) + max(g[i + 1, j - 1, 0],
                                                                                   g[i + 1, j - 1, 1]) + max(
                                                                                   g[i + 2, j, 0], g[i + 2, j, 1])
    # case: edges from t' to t: 1-0 or 1-1 (same as t->t': 0-1 or 1-1) - looking for edges from t' to t'' 1-0 or 1-1
    if n_ed == 1 or n_ed == 3:
        summa += max(g[i, j, 0], g[i, j, 2])
        # horiz edges
        if i % 2 == 0:
            if i == 0:
                if j % 2 == 1:
                    if j == g.shape[1] - 2:
                        summa += max(g[i + 1, j + 1, 2], g[i + 1, j + 1, 3])
                    else:
                        summa += max(g[i + 1, j + 1, 2], g[i + 1, j + 1, 3]) + max(g[i, j + 2, 2], g[i, j + 2, 3])
            elif i == g.shape[0] - 1:
                if j % 2 == 1:
                    if j == g.shape[1] - 2:
                        summa += max(g[i - 1, j + 1, 2], g[i - 1, j + 1, 3])
                    else:
                        summa += max(g[i - 1, j + 1, 2], g[i - 1, j + 1, 3]) + max(g[i, j + 2, 2], g[i, j + 2, 3])
            else:
                if j % 2 == 1:
                    if j == g.shape[1] - 2:
                        summa += max(g[i - 1, j + 1, 2], g[i - 1, j + 1, 3]) + max(g[i + 1, j + 1, 2],
                                                                                   g[i + 1, j + 1, 3])
                    else:
                        summa += max(g[i - 1, j + 1, 2], g[i - 1, j + 1, 3]) + max(g[i + 1, j + 1, 2],
                                                                                   g[i + 1, j + 1, 3]) + max(
                                                                                   g[i, j + 2, 2], g[i, j + 2, 3])
        # vert edges
        elif i % 2 == 1:
            if i == g.shape[0] - 2:
                if j % 2 == 0:
                    if j == 0:
                        summa += max(g[i + 1, j + 1, 2], g[i + 1, j + 1, 3])
                    elif j == g.shape[1] - 1:
                        summa += max(g[i + 1, j - 1, 2], g[i + 1, j - 1, 3])
                    else:
                        summa += max(g[i + 1, j + 1, 2], g[i + 1, j + 1, 3]) + max(g[i + 1, j - 1, 2],
                                                                                   g[i + 1, j - 1, 3])
            else:
                if j % 2 == 0:
                    if j == 0:
                        summa += max(g[i + 1, j + 1, 2], g[i + 1, j + 1, 3]) + max(g[i + 2, j, 2], g[i + 2, j, 3])
                    elif j == g.shape[1] - 1:
                        summa += max(g[i + 1, j - 1, 2], g[i + 1, j - 1, 3]) + max(g[i + 2, j, 2], g[i + 2, j, 3])
                    else:
                        summa += max(g[i + 1, j + 1, 2], g[i + 1, j + 1, 3]) + max(g[i + 1, j - 1, 2],
                                                                                   g[i + 1, j - 1, 3]) + max(
                                                                                   g[i + 2, j, 2], g[i + 2, j, 3])
    # end t' -> t

    return summa / n_nei


# !!! ONE iteration of diffusion: next "edge-update" is not depends from neighbour edge that was updated on current
# iteration of diffusion algorithm (if must depends => change all "new_graph_edges" to "graph_edges" !!!
@njit(fastmath=True)
def diffusion_iter(graph_edges):
    new_graph_edges = np.zeros_like(graph_edges)
    for i in range(graph_edges.shape[0]):
        for j in range(graph_edges.shape[1]):
            if (i + j) % 2 == 0:
                continue
            new_graph_edges[i, j, 0] = graph_edges[i, j, 0] - max(graph_edges[i, j, 0], graph_edges[i, j, 1]) - max(
                graph_edges[i, j, 0], graph_edges[i, j, 2]) + big_sum(i, j, 0, graph_edges)
            new_graph_edges[i, j, 1] = graph_edges[i, j, 1] - max(graph_edges[i, j, 1], graph_edges[i, j, 0]) - max(
                graph_edges[i, j, 1], graph_edges[i, j, 3]) + big_sum(i, j, 1, graph_edges)
            new_graph_edges[i, j, 2] = graph_edges[i, j, 2] - max(graph_edges[i, j, 2], graph_edges[i, j, 3]) - max(
                graph_edges[i, j, 2], graph_edges[i, j, 0]) + big_sum(i, j, 2, graph_edges)
            new_graph_edges[i, j, 3] = graph_edges[i, j, 3] - max(graph_edges[i, j, 3], graph_edges[i, j, 2]) - max(
                graph_edges[i, j, 3], graph_edges[i, j, 1]) + big_sum(i, j, 3, graph_edges)
    return new_graph_edges


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
filename = "field6.jpg"
img1 = imread(filename)
scale = [140, 90]
n_iterations = 1000
eps = 0.1
img1_res = resize(img1, scale)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

start_time = time.time()

mean0, mean1, cov0, cov1 = take_params(img1_res)
PROBS = get_probs(img1_res, mean0, mean1, cov0, cov1)
graph = fill_graph_edges(PROBS)

zero_iter_time = time.time() - start_time
print("Time of zero iteration (get probs and make start graph):  %s seconds; " % zero_iter_time)

start_time = time.time()

print(graph[:5, 0])
iteration = 0
while iteration < n_iterations:
    iteration += 1
    # print(iteration)
    graph = diffusion_iter(graph)
    # print(graph[:5, 0])
print(graph[:5, 0])

alg_time = time.time() - start_time
print("Diffusion time: %s seconds;" % alg_time)
print("time per diffusion iteration: ", alg_time / n_iterations)
# show_image(img1_res, size=(9, 7))
