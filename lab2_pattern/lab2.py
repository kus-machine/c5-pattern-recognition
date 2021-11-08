from numba import njit, prange
import numpy as np
from cv2 import COLOR_BGR2RGB, cvtColor
import matplotlib.pyplot as plt


# func that show the image (put in BGR (cv2))
def show_image(image, size=(9, 7)):
    plt.figure(figsize=size)
    # Before showing image, bgr color order transformed to rgb order
    plt.imshow(cvtColor(image, COLOR_BGR2RGB))
    plt.xticks([])
    plt.yticks([])
    plt.show()


# take parameters of multi-gauss distribution of 2 classes (sky and ground) for good axis
def take_params(image, h=.2, w=.4, offset=0.05):
    if w > 0.5:
        w = 0.5
    piece0 = image[int(offset * image.shape[0]):int((h + offset) * image.shape[0]),
             int(image.shape[1] * (1 / 2 - w)):int(
                 image.shape[1] * (1 / 2 + w))]
    piece1 = image[int((1 - h - offset) * image.shape[0]):int((1 - offset) * image.shape[0]), int(image.shape[1] * (
            1 / 2 - w)):int(image.shape[1] * (1 / 2 + w))]

    piece0 = np.reshape(piece0, (piece0.shape[0] * piece0.shape[1], piece0.shape[2]))
    piece1 = np.reshape(piece1, (piece1.shape[0] * piece1.shape[1], piece1.shape[2]))

    mean_0 = np.mean(piece0, axis=0)
    mean_1 = np.mean(piece1, axis=0)

    cov_0 = np.cov([piece0[..., 0], piece0[..., 1], piece0[..., 2]])
    cov_1 = np.cov([piece1[..., 0], piece1[..., 1], piece1[..., 2]])

    return mean_0, mean_1, cov_0, cov_1


# return logarithm of multi-gauss distribution (simplified on sheet of paper)
# @njit#(fastmath=True)
def pdf_multivariate_gauss_log(x, mu, cov):
    # part1 = 1 / (((2 * np.pi) ** (mu.shape[0] / 2)) * (np.linalg.det(cov) ** (1 / 2)))
    # part2 = (-1 / 2) * (((x - mu).T).dot(np.linalg.inv(cov))).dot((x - mu))
    # return np.double(part1 * np.exp(part2))
    # _____________________________________________________________________
    # take log() from gauss distr: log(exp(blablabla)/Z) = blablabla - log(Z)
    cov = cov.astype(float)
    part1 = np.log(np.linalg.det(cov))
    part2 = (-1 / 2) * ((x - mu).T.dot(np.linalg.inv(cov))).dot((x - mu))
    return part2 - 2 * part1


# @njit(cache=False, nogil=False, fastmath=True, parallel=True)
# @njit(fastmath=True, parallel=True)
def get_probs(image, mean_0, mean_1, cov_0, cov_1):
    image_resh = image.reshape(image.shape[0] * image.shape[1], image.shape[2])
    PROB = np.zeros((image_resh.shape[0], 2), float)
    for i in prange(image_resh.shape[0]):
        # for j in prange(image.shape[1]-1):
        PROB[i, 0] = pdf_multivariate_gauss_log(image_resh[i], mean_0, cov_0)
        PROB[i, 1] = pdf_multivariate_gauss_log(image_resh[i], mean_1, cov_1)
    return PROB.reshape((image.shape[0], image.shape[1], 2))


# this func only for 1st iteration (fill start graph)
@njit(fastmath=True)
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


# @njit(fastmath=True)
def fill_graph_edges(probs, epsilon=0.25):
    g_for_im = np.zeros((2 * probs.shape[0] - 1, 2 * probs.shape[1] - 1, 4), float)
    # 4 edges - 0-0, 0-1, 1-0, 1-1 in this order
    for i in range(g_for_im.shape[0]):
        # четные i; смотрим объекты слева и справа
        if i % 2 == 0:
            for j in range(1, g_for_im.shape[1] - 1, 2):
                g_for_im[i, j, 0] = (probs[i // 2, (j - 1) // 2, 0] + probs[i // 2, (j + 1) // 2, 0]) / n_of_Nt(
                    i, j, g_for_im.shape) + 2 * np.log(epsilon)
                g_for_im[i, j, 1] = (probs[i // 2, (j - 1) // 2, 0] + probs[i // 2, (j + 1) // 2, 1]) / n_of_Nt(
                    i, j, g_for_im.shape) + 2 * np.log(1 - epsilon)
                g_for_im[i, j, 2] = (probs[i // 2, (j - 1) // 2, 1] + probs[i // 2, (j + 1) // 2, 0]) / n_of_Nt(
                    i, j, g_for_im.shape) + 2 * np.log(1 - epsilon)
                g_for_im[i, j, 3] = (probs[i // 2, (j - 1) // 2, 1] + probs[i // 2, (j + 1) // 2, 1]) / n_of_Nt(
                    i, j, g_for_im.shape) + 2 * np.log(epsilon)
        # нечетные i; смотрим объекты сверху и снизу
        else:
            for j in range(0, g_for_im.shape[1], 2):
                g_for_im[i, j, 0] = (probs[(i - 1) // 2, j // 2, 0] + probs[(i + 1) // 2, j // 2, 0]) / n_of_Nt(
                    i, j, g_for_im.shape) + 2 * np.log(epsilon)
                g_for_im[i, j, 1] = (probs[(i - 1) // 2, j // 2, 0] + probs[(i + 1) // 2, j // 2, 1]) / n_of_Nt(
                    i, j, g_for_im.shape) + 2 * np.log(1 - epsilon)
                g_for_im[i, j, 2] = (probs[(i - 1) // 2, j // 2, 1] + probs[(i + 1) // 2, j // 2, 0]) / n_of_Nt(
                    i, j, g_for_im.shape) + 2 * np.log(1 - epsilon)
                g_for_im[i, j, 3] = (probs[(i - 1) // 2, j // 2, 1] + probs[(i + 1) // 2, j // 2, 1]) / n_of_Nt(
                    i, j, g_for_im.shape) + 2 * np.log(epsilon)
    return g_for_im


@njit(fastmath=True)
def big_sum(i, j, n_ed, g, time):
    summa = 0.
    n_nei = 8.
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
    if time == 1:
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
                            summa += max(g[i + 1, j - 1, 0], g[i + 1, j - 1, 1]) + max(g[i, j - 2, 0], g[i, j - 2, 2])
                elif i == g.shape[0] - 1:
                    if j % 2 == 1:
                        if j == 1:
                            summa += max(g[i - 1, j - 1, 0], g[i - 1, j - 1, 2])
                        else:
                            summa += max(g[i - 1, j - 1, 0], g[i - 1, j - 1, 2]) + max(g[i, j - 2, 0], g[i, j - 2, 2])
                else:
                    if j % 2 == 1:
                        if j == 1:
                            summa += max(g[i - 1, j - 1, 0], g[i - 1, j - 1, 2]) + max(g[i + 1, j - 1, 0],
                                                                                       g[i + 1, j - 1, 1])
                        else:
                            summa += max(g[i - 1, j - 1, 0], g[i - 1, j - 1, 2]) + max(g[i + 1, j - 1, 0],
                                                                                       g[i + 1, j - 1, 1]) + max(
                                g[i, j - 2, 0], g[i, j - 2, 2])
            # vert edges - watch for t - top
            elif i % 2 == 1:
                if i == 1:
                    if j % 2 == 0:
                        if j == 0:
                            summa += max(g[i - 1, j + 1, 0], g[i - 1, j + 1, 1])
                        elif j == g.shape[1] - 1:
                            summa += max(g[i - 1, j - 1, 0], g[i - 1, j - 1, 2])
                        else:
                            summa += max(g[i - 1, j - 1, 0], g[i - 1, j - 1, 2]) + max(g[i - 1, j + 1, 0],
                                                                                       g[i - 1, j + 1, 1])
                else:
                    if j % 2 == 0:
                        if j == 0:
                            summa += max(g[i - 1, j + 1, 0], g[i - 1, j + 1, 1]) + max(g[i - 2, j, 0], g[i - 2, j, 2])
                        elif j == g.shape[1] - 1:
                            summa += max(g[i - 1, j - 1, 0], g[i - 1, j - 1, 2]) + max(g[i - 2, j, 0], g[i - 2, j, 2])
                        else:
                            summa += max(g[i - 1, j + 1, 0], g[i - 1, j + 1, 1]) + max(g[i - 1, j - 1, 0],
                                                                                       g[i - 1, j - 1, 2]) + max(
                                g[i - 2, j, 0], g[i - 2, j, 2])
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
                            summa += max(g[i + 1, j - 1, 2], g[i + 1, j - 1, 3]) + max(g[i, j - 2, 1], g[i, j - 2, 3])
                elif i == g.shape[0] - 1:
                    if j % 2 == 1:
                        if j == 1:
                            summa += max(g[i - 1, j - 1, 1], g[i - 1, j - 1, 3])
                        else:
                            summa += max(g[i - 1, j - 1, 1], g[i - 1, j - 1, 3]) + max(g[i, j - 2, 1], g[i, j - 2, 3])
                else:
                    if j % 2 == 1:
                        if j == 1:
                            summa += max(g[i - 1, j - 1, 1], g[i - 1, j - 1, 3]) + max(g[i + 1, j - 1, 2],
                                                                                       g[i + 1, j - 1, 3])
                        else:
                            summa += max(g[i - 1, j - 1, 1], g[i - 1, j - 1, 3]) + max(g[i + 1, j - 1, 2],
                                                                                       g[i + 1, j - 1, 3]) + max(
                                g[i, j - 2, 1],
                                g[i, j - 2, 3])
            # vert edges - watch for t - top
            elif i % 2 == 1:
                if i == 1:
                    if j % 2 == 0:
                        if j == 0:
                            summa += max(g[i - 1, j + 1, 2], g[i - 1, j + 1, 3])
                        elif j == g.shape[1] - 1:
                            summa += max(g[i - 1, j - 1, 1], g[i - 1, j - 1, 3])
                        else:
                            summa += max(g[i - 1, j - 1, 1], g[i - 1, j - 1, 3]) + max(g[i - 1, j + 1, 2],
                                                                                       g[i - 1, j + 1, 3])
                else:
                    if j % 2 == 0:
                        if j == 0:
                            summa += max(g[i - 1, j + 1, 2], g[i - 1, j + 1, 3]) + max(g[i - 2, j, 1], g[i - 2, j, 3])
                        elif j == g.shape[1] - 1:
                            summa += max(g[i - 1, j - 1, 1], g[i - 1, j - 1, 3]) + max(g[i - 2, j, 1], g[i - 2, j, 3])
                        else:
                            summa += max(g[i - 1, j + 1, 2], g[i - 1, j + 1, 3]) + max(g[i - 1, j - 1, 1],
                                                                                       g[i - 1, j - 1, 3]) + max(
                                g[i - 2, j, 1],
                                g[i - 2, j, 3])
        # end t -> t'
        # -----------------------------------------------------------------------------------------------------------------
    # start t' -> t
    # now, calculate second side of edge g(i,j): from t' to t (we think that current object is t')
    elif time == 2:
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
                            summa += max(g[i - 1, j + 1, 0], g[i - 1, j + 1, 2])
                        else:
                            summa += max(g[i - 1, j + 1, 0], g[i - 1, j + 1, 2]) + max(g[i, j + 2, 0], g[i, j + 2, 1])
                else:
                    if j % 2 == 1:
                        if j == g.shape[1] - 2:
                            summa += max(g[i - 1, j + 1, 0], g[i - 1, j + 1, 2]) + max(g[i + 1, j + 1, 0],
                                                                                       g[i + 1, j + 1, 1])
                        else:
                            summa += max(g[i - 1, j + 1, 0], g[i - 1, j + 1, 2]) + max(g[i + 1, j + 1, 0],
                                                                                       g[i + 1, j + 1, 1]) + max(
                                g[i, j + 2, 0], g[i, j + 2, 1])
            # vert edges
            elif i % 2 == 1:
                if i == g.shape[0] - 2:
                    if j % 2 == 0:
                        if j == 0:
                            summa += max(g[i + 1, j + 1, 0], g[i + 1, j + 1, 1])
                        elif j == g.shape[1] - 1:
                            summa += max(g[i + 1, j - 1, 0], g[i + 1, j - 1, 2])
                        else:
                            summa += max(g[i + 1, j + 1, 0], g[i + 1, j + 1, 1]) + max(g[i + 1, j - 1, 0],
                                                                                       g[i + 1, j - 1, 2])
                else:
                    if j % 2 == 0:
                        if j == 0:
                            summa += max(g[i + 1, j + 1, 0], g[i + 1, j + 1, 1]) + max(g[i + 2, j, 0], g[i + 2, j, 1])
                        elif j == g.shape[1] - 1:
                            summa += max(g[i + 1, j - 1, 0], g[i + 1, j - 1, 2]) + max(g[i + 2, j, 0], g[i + 2, j, 1])
                        else:
                            summa += max(g[i + 1, j + 1, 0], g[i + 1, j + 1, 1]) + max(g[i + 1, j - 1, 0],
                                                                                       g[i + 1, j - 1, 2]) + max(
                                g[i + 2, j, 0], g[i + 2, j, 1])
        # case: edges from t' to t: 1-0 or 1-1 (same as t->t': 0-1 or 1-1) - looking for edges from t' to t'' 1-0 or 1-1
        elif n_ed == 1 or n_ed == 3:
            summa += max(g[i, j, 1], g[i, j, 3])
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
                            summa += max(g[i - 1, j + 1, 1], g[i - 1, j + 1, 3])
                        else:
                            summa += max(g[i - 1, j + 1, 1], g[i - 1, j + 1, 3]) + max(g[i, j + 2, 2], g[i, j + 2, 3])
                else:
                    if j % 2 == 1:
                        if j == g.shape[1] - 2:
                            summa += max(g[i - 1, j + 1, 1], g[i - 1, j + 1, 3]) + max(g[i + 1, j + 1, 2],
                                                                                       g[i + 1, j + 1, 3])
                        else:
                            summa += max(g[i - 1, j + 1, 1], g[i - 1, j + 1, 3]) + max(g[i + 1, j + 1, 2],
                                                                                       g[i + 1, j + 1, 3]) + max(
                                g[i, j + 2, 2], g[i, j + 2, 3])
            # vert edges
            elif i % 2 == 1:
                if i == g.shape[0] - 2:
                    if j % 2 == 0:
                        if j == 0:
                            summa += max(g[i + 1, j + 1, 2], g[i + 1, j + 1, 3])
                        elif j == g.shape[1] - 1:
                            summa += max(g[i + 1, j - 1, 1], g[i + 1, j - 1, 3])
                        else:
                            summa += max(g[i + 1, j + 1, 2], g[i + 1, j + 1, 3]) + max(g[i + 1, j - 1, 1],
                                                                                       g[i + 1, j - 1, 3])
                else:
                    if j % 2 == 0:
                        if j == 0:
                            summa += max(g[i + 1, j + 1, 2], g[i + 1, j + 1, 3]) + max(g[i + 2, j, 2], g[i + 2, j, 3])
                        elif j == g.shape[1] - 1:
                            summa += max(g[i + 1, j - 1, 1], g[i + 1, j - 1, 3]) + max(g[i + 2, j, 2], g[i + 2, j, 3])
                        else:
                            summa += max(g[i + 1, j + 1, 2], g[i + 1, j + 1, 3]) + max(g[i + 1, j - 1, 1],
                                                                                       g[i + 1, j - 1, 3]) + max(
                                g[i + 2, j, 2], g[i + 2, j, 3])
        # end t' -> t

    return summa # / n_nei


# !!! ONE iteration of diffusion: next "edge-update" is not depends from neighbour edge that was updated on current
# iteration of diffusion algorithm (if must depends => change all "new_graph_edges" to "graph_edges" !!!
@njit(fastmath=True, parallel=True)
def diffusion_iter(graph_edges_for_diffusion_iter):
    graph_ed = graph_edges_for_diffusion_iter.copy()
    # new_graph_edges = np.zeros_like(graph_edges, float)
    for i in prange(graph_ed.shape[0]):
        for j in range(graph_ed.shape[1]):
            if (i + j) % 2 != 0:
                # continue
                graph_ed[i, j, 0] += - max(graph_ed[i, j, 0], graph_ed[i, j, 1]) + big_sum(i, j, 0, graph_ed, 1) / n_of_Nt(i, j, graph_ed.shape)
                graph_ed[i, j, 1] += - max(graph_ed[i, j, 1], graph_ed[i, j, 0]) + big_sum(i, j, 1, graph_ed, 1) / n_of_Nt(i, j, graph_ed.shape)
                graph_ed[i, j, 2] += - max(graph_ed[i, j, 2], graph_ed[i, j, 3]) + big_sum(i, j, 2, graph_ed, 1) / n_of_Nt(i, j, graph_ed.shape)
                graph_ed[i, j, 3] += - max(graph_ed[i, j, 3], graph_ed[i, j, 2]) + big_sum(i, j, 3, graph_ed, 1) / n_of_Nt(i, j, graph_ed.shape)
                graph_ed[i, j, 0] += - max(graph_ed[i, j, 0], graph_ed[i, j, 2]) + big_sum(i, j, 0, graph_ed, 2) / n_of_Nt(i, j, graph_ed.shape)
                graph_ed[i, j, 1] += - max(graph_ed[i, j, 1], graph_ed[i, j, 3]) + big_sum(i, j, 1, graph_ed, 2) / n_of_Nt(i, j, graph_ed.shape)
                graph_ed[i, j, 2] += - max(graph_ed[i, j, 2], graph_ed[i, j, 0]) + big_sum(i, j, 2, graph_ed, 2) / n_of_Nt(i, j, graph_ed.shape)
                graph_ed[i, j, 3] += - max(graph_ed[i, j, 3], graph_ed[i, j, 1]) + big_sum(i, j, 3, graph_ed, 2) / n_of_Nt(i, j, graph_ed.shape)
    return graph_ed


@njit(fastmath=True)
def diffusion_alg(n_iter, graph):
    for i in range(n_iter):
        graph = diffusion_iter(graph)
    return graph


@njit(fastmath=True)
def deletion_iter(graph_for_del_iter, epsilon):
    g = graph_for_del_iter.copy()
    # убираем все дуги, отличающиеся более чем на epsilon от максимальной между парой объектов
    for i, j in np.ndindex(g.shape[:2]):
        if (i + j) % 2 != 0:
            local_max = max(g[i, j])
            for k in range(4):
                if g[i, j, k] + epsilon < local_max:
                    g[i, j, k] = -np.inf
    for i, j in np.ndindex(g.shape[:2]):
        # из нуля не торчит в след
        if g[i, j, 0] == -np.inf and g[i, j, 1] == -np.inf:
            # horiz edges - watch for t - left
            if i % 2 == 0:
                if i == 0:
                    if j % 2 == 1:
                        if j == 1:
                            g[i + 1, j - 1, 0] = g[i + 1, j - 1, 1] = -np.inf
                        else:
                            g[i + 1, j - 1, 0] = g[i + 1, j - 1, 1] = g[i, j - 2, 0] = g[i, j - 2, 2] = -np.inf
                elif i == g.shape[0] - 1:
                    if j % 2 == 1:
                        if j == 1:
                            g[i - 1, j - 1, 0] = g[i - 1, j - 1, 1] = -np.inf
                        else:
                            g[i - 1, j - 1, 0] = g[i - 1, j - 1, 1] = g[i, j - 2, 0] = g[i, j - 2, 2] = -np.inf
                else:
                    if j % 2 == 1:
                        if j == 1:
                            g[i - 1, j - 1, 0] = g[i - 1, j - 1, 2] = g[i + 1, j - 1, 0] = g[i + 1, j - 1, 1] = \
                                -np.inf
                        else:
                            g[i - 1, j - 1, 0] = g[i - 1, j - 1, 2] = g[i + 1, j - 1, 0] = g[i + 1, j - 1, 1] = \
                                g[i, j - 2, 0] = g[i, j - 2, 2] = -np.inf
            # vert edges - watch for t - top
            elif i % 2 == 1:
                if i == 1:
                    if j % 2 == 0:
                        if j == 0:
                            g[i - 1, j + 1, 0] = g[i - 1, j + 1, 1] = -np.inf
                        elif j == g.shape[1] - 1:
                            g[i - 1, j - 1, 0] = g[i - 1, j - 1, 2] = -np.inf
                        else:
                            g[i - 1, j - 1, 0] = g[i - 1, j - 1, 2] = g[i - 1, j + 1, 0] = g[i - 1, j + 1, 1] = \
                                -np.inf
                else:
                    if j % 2 == 0:
                        if j == 0:
                            g[i - 1, j + 1, 0] = g[i - 1, j + 1, 1] = g[i - 2, j, 0] = g[i - 2, j, 2] = -np.inf
                        elif j == g.shape[1] - 1:
                            g[i - 1, j - 1, 0] = g[i - 1, j - 1, 2] = g[i - 2, j, 0] = g[i - 2, j, 2] = -np.inf
                        else:
                            g[i - 1, j + 1, 0] = g[i - 1, j + 1, 1] = g[i - 1, j - 1, 0] = g[i - 1, j - 1, 2] = \
                                g[i - 2, j, 0] = g[i - 2, j, 2] = -np.inf

        # из нуля не торчит в пред
        if g[i, j, 0] == -np.inf and g[i, j, 2] == -np.inf:
            if i % 2 == 0:
                if i == 0:
                    if j % 2 == 1:
                        if j == g.shape[1] - 2:
                            g[i + 1, j + 1, 0] = g[i + 1, j + 1, 1] = -np.inf
                        else:
                            g[i + 1, j + 1, 0] = g[i + 1, j + 1, 1] = g[i, j + 2, 0] = g[i, j + 2, 1] = -np.inf
                elif i == g.shape[0] - 1:
                    if j % 2 == 1:
                        if j == g.shape[1] - 2:
                            g[i - 1, j + 1, 0] = g[i - 1, j + 1, 2] = -np.inf
                        else:
                            g[i - 1, j + 1, 0] = g[i - 1, j + 1, 2] = g[i, j + 2, 0] = g[i, j + 2, 1] = -np.inf
                else:
                    if j % 2 == 1:
                        if j == g.shape[1] - 2:
                            g[i - 1, j + 1, 0] = g[i - 1, j + 1, 2] = g[i + 1, j + 1, 0] = g[i + 1, j + 1, 1] = \
                                -np.inf
                        else:
                            g[i - 1, j + 1, 0] = g[i - 1, j + 1, 2] = g[i + 1, j + 1, 0] = g[i + 1, j + 1, 1] = \
                                g[i, j + 2, 0] = g[i, j + 2, 1] = -np.inf
            # vert edges
            elif i % 2 == 1:
                if i == g.shape[0] - 2:
                    if j % 2 == 0:
                        if j == 0:
                            g[i + 1, j + 1, 0] = g[i + 1, j + 1, 1] = -np.inf
                        elif j == g.shape[1] - 1:
                            g[i + 1, j - 1, 0] = g[i + 1, j - 1, 2] = -np.inf
                        else:
                            g[i + 1, j + 1, 0] = g[i + 1, j + 1, 1] = g[i + 1, j - 1, 0] = g[i + 1, j - 1, 2] = \
                                -np.inf
                else:
                    if j % 2 == 0:
                        if j == 0:
                            g[i + 1, j + 1, 0] = g[i + 1, j + 1, 1] = g[i + 2, j, 0] = g[i + 2, j, 1] = -np.inf
                        elif j == g.shape[1] - 1:
                            g[i + 1, j - 1, 0] = g[i + 1, j - 1, 2] = g[i + 2, j, 0] = g[i + 2, j, 1] = -np.inf
                        else:
                            g[i + 1, j + 1, 0] = g[i + 1, j + 1, 1] = g[i + 1, j - 1, 0] = g[i + 1, j - 1, 2] = \
                                g[i + 2, j, 0] = g[i + 2, j, 1] = -np.inf

        # из еденицы не торчит в пред
        if g[i, j, 1] == -np.inf and g[i, j, 3] == -np.inf:
            if i % 2 == 0:
                if i == 0:
                    if j % 2 == 1:
                        if j == g.shape[1] - 2:
                            g[i + 1, j + 1, 2] = g[i + 1, j + 1, 3] = -np.inf
                        else:
                            g[i + 1, j + 1, 2] = g[i + 1, j + 1, 3] = g[i, j + 2, 2] = g[i, j + 2, 3] = -np.inf
                elif i == g.shape[0] - 1:
                    if j % 2 == 1:
                        if j == g.shape[1] - 2:
                            g[i - 1, j + 1, 1] = g[i - 1, j + 1, 3] = -np.inf
                        else:
                            g[i - 1, j + 1, 1] = g[i - 1, j + 1, 3] = g[i, j + 2, 2] = g[i, j + 2, 3] = -np.inf
                else:
                    if j % 2 == 1:
                        if j == g.shape[1] - 2:
                            g[i - 1, j + 1, 1] = g[i - 1, j + 1, 3] = g[i + 1, j + 1, 2] = g[i + 1, j + 1, 3] = \
                                -np.inf
                        else:
                            g[i - 1, j + 1, 1] = g[i - 1, j + 1, 3] = g[i + 1, j + 1, 2] = g[i + 1, j + 1, 3] = \
                                g[i, j + 2, 2] = g[i, j + 2, 3] = -np.inf
            # vert edges
            elif i % 2 == 1:
                if i == g.shape[0] - 2:
                    if j % 2 == 0:
                        if j == 0:
                            g[i + 1, j + 1, 2] = g[i + 1, j + 1, 3] = -np.inf
                        elif j == g.shape[1] - 1:
                            g[i + 1, j - 1, 1] = g[i + 1, j - 1, 3] = -np.inf
                        else:
                            g[i + 1, j + 1, 2] = g[i + 1, j + 1, 3] = g[i + 1, j - 1, 1] = g[i + 1, j - 1, 3] = \
                                -np.inf
                else:
                    if j % 2 == 0:
                        if j == 0:
                            g[i + 1, j + 1, 2] = g[i + 1, j + 1, 3] = g[i + 2, j, 2] = g[i + 2, j, 3] = -np.inf
                        elif j == g.shape[1] - 1:
                            g[i + 1, j - 1, 1] = g[i + 1, j - 1, 3] = g[i + 2, j, 2] = g[i + 2, j, 3] = -np.inf
                        else:
                            g[i + 1, j + 1, 2] = g[i + 1, j + 1, 3] = g[i + 1, j - 1, 1] = g[i + 1, j - 1, 3] = \
                                g[i + 2, j, 2] = g[i + 2, j, 3] = -np.inf

        # из единицы не торчит в след
        if g[i, j, 2] == -np.inf and g[i, j, 3] == -np.inf:
            if i % 2 == 0:
                if i == 0:
                    if j % 2 == 1:
                        if j == 1:
                            g[i + 1, j - 1, 2] = g[i + 1, j - 1, 3] = -np.inf
                        else:
                            g[i + 1, j - 1, 2] = g[i + 1, j - 1, 3] = g[i, j - 2, 1] = g[i, j - 2, 3] = -np.inf
                elif i == g.shape[0] - 1:
                    if j % 2 == 1:
                        if j == 1:
                            g[i - 1, j - 1, 1] = g[i - 1, j - 1, 3] = -np.inf
                        else:
                            g[i - 1, j - 1, 1] = g[i - 1, j - 1, 3] = g[i, j - 2, 1] = g[i, j - 2, 3] = -np.inf
                else:
                    if j % 2 == 1:
                        if j == 1:
                            g[i - 1, j - 1, 1] = g[i - 1, j - 1, 3] = g[i + 1, j - 1, 2] = g[i + 1, j - 1, 3] = \
                                -np.inf
                        else:
                            g[i - 1, j - 1, 1] = g[i - 1, j - 1, 3] = g[i + 1, j - 1, 2] = g[i + 1, j - 1, 3] = \
                                g[i, j - 2, 1] = g[i, j - 2, 3] = -np.inf
            # vert edges - watch for t - top
            elif i % 2 == 1:
                if i == 1:
                    if j % 2 == 0:
                        if j == 0:
                            g[i - 1, j + 1, 2] = g[i - 1, j + 1, 3] = -np.inf
                        elif j == g.shape[1] - 1:
                            g[i - 1, j - 1, 1] = g[i - 1, j - 1, 3] = -np.inf
                        else:
                            g[i - 1, j - 1, 1] = g[i - 1, j - 1, 3] = g[i - 1, j + 1, 2] = g[i - 1, j + 1, 3] = \
                                -np.inf
                else:
                    if j % 2 == 0:
                        if j == 0:
                            g[i - 1, j + 1, 2] = g[i - 1, j + 1, 3] = g[i - 2, j, 1] = g[i - 2, j, 3] = -np.inf
                        elif j == g.shape[1] - 1:
                            g[i - 1, j - 1, 1] = g[i - 1, j - 1, 3] = g[i - 2, j, 1] = g[i - 2, j, 3] = -np.inf
                        else:
                            g[i - 1, j + 1, 2] = g[i - 1, j + 1, 3] = g[i - 1, j - 1, 1] = g[i - 1, j - 1, 3] = \
                                g[i - 2, j, 1] = g[i - 2, j, 3] = -np.inf
    return g


@njit(fastmath=True)
def deletion_alg(gr, epsilon):
    gr1 = gr.copy()
    while True:
        new_gr = deletion_iter(gr1, epsilon)
        if np.all(new_gr == gr1):
            # print("Deletions algorithm completed")
            break
        else:
            gr1 = new_gr.copy()
    return new_gr


@njit(fastmath=True)
def try_build_labeling(gr, labels):
    image_is_good = True
    labels[..., :] = 0
    for i, j in np.ndindex(gr.shape[:2]):
        if image_is_good:
            if gr[i, j].max() == -np.inf:
                image_is_good = False
                return image_is_good, labels
    for i, j in np.ndindex(labels.shape[:2]):
        if i == 0:
            if j == 0:
                class0 = max(gr[2 * i + 1, 2 * j, 0], gr[2 * i + 1, 2 * j, 1], gr[2 * i, 2 * j + 1, 0],
                             gr[2 * i, 2 * j + 1, 1])
                class1 = max(gr[2 * i + 1, 2 * j, 2], gr[2 * i + 1, 2 * j, 3], gr[2 * i, 2 * j + 1, 2],
                             gr[2 * i, 2 * j + 1, 3])
                if class1 > class0:
                    labels[i, j] = 1
            elif j == labels.shape[1] - 1:
                class0 = max(gr[2 * i + 1, 2 * j, 0], gr[2 * i + 1, 2 * j, 1], gr[2 * i, 2 * j - 1, 0],
                             gr[2 * i, 2 * j - 1, 2])
                class1 = max(gr[2 * i + 1, 2 * j, 2], gr[2 * i + 1, 2 * j, 3], gr[2 * i, 2 * j - 1, 1],
                             gr[2 * i, 2 * j - 1, 3])
                if class1 > class0:
                    labels[i, j] = 1
            else:
                class0 = max(gr[2 * i + 1, 2 * j, 0], gr[2 * i + 1, 2 * j, 1], gr[2 * i, 2 * j + 1, 0],
                             gr[2 * i, 2 * j + 1, 1], gr[2 * i, 2 * j - 1, 0], gr[2 * i, 2 * j - 1, 2])
                class1 = max(gr[2 * i + 1, 2 * j, 2], gr[2 * i + 1, 2 * j, 3], gr[2 * i, 2 * j + 1, 2],
                             gr[2 * i, 2 * j + 1, 3], gr[2 * i, 2 * j - 1, 1], gr[2 * i, 2 * j - 1, 3])
                if class1 > class0:
                    labels[i, j] = 1
        elif i == labels.shape[0] - 1:
            if j == 0:
                class0 = max(gr[2 * i, 2 * j + 1, 0], gr[2 * i, 2 * j + 1, 1], gr[2 * i - 1, 2 * j, 0],
                             gr[2 * i - 1, 2 * j, 2])
                class1 = max(gr[2 * i, 2 * j + 1, 2], gr[2 * i, 2 * j + 1, 3], gr[2 * i - 1, 2 * j, 1],
                             gr[2 * i - 1, 2 * j, 3])
                if class1 > class0:
                    labels[i, j] = 1
            elif j == labels.shape[1] - 1:
                class0 = max(gr[2 * i - 1, 2 * j, 0], gr[2 * i - 1, 2 * j, 2], gr[2 * i, 2 * j - 1, 0],
                             gr[2 * i, 2 * j - 1, 2])
                class1 = max(gr[2 * i - 1, 2 * j, 1], gr[2 * i - 1, 2 * j, 3], gr[2 * i, 2 * j - 1, 1],
                             gr[2 * i, 2 * j - 1, 3])
                if class1 > class0:
                    labels[i, j] = 1
            else:
                class0 = max(gr[2 * i, 2 * j + 1, 0], gr[2 * i, 2 * j + 1, 1], gr[2 * i - 1, 2 * j, 0],
                             gr[2 * i - 1, 2 * j, 2], gr[2 * i, 2 * j - 1, 0],
                             gr[2 * i, 2 * j - 1, 2])
                class1 = max(gr[2 * i, 2 * j + 1, 2], gr[2 * i, 2 * j + 1, 3], gr[2 * i - 1, 2 * j, 1],
                             gr[2 * i - 1, 2 * j, 3], gr[2 * i, 2 * j - 1, 1],
                             gr[2 * i, 2 * j - 1, 3])
                if class1 > class0:
                    labels[i, j] = 1
        else:
            if j == 0:
                class0 = max(gr[2 * i + 1, 2 * j, 0], gr[2 * i + 1, 2 * j, 1], gr[2 * i, 2 * j + 1, 0],
                             gr[2 * i, 2 * j + 1, 1], gr[2 * i - 1, 2 * j, 0], gr[2 * i - 1, 2 * j, 2])
                class1 = max(gr[2 * i + 1, 2 * j, 2], gr[2 * i + 1, 2 * j, 3], gr[2 * i, 2 * j + 1, 2],
                             gr[2 * i, 2 * j + 1, 3], gr[2 * i - 1, 2 * j, 1], gr[2 * i - 1, 2 * j, 3], )
                if class1 > class0:
                    labels[i, j] = 1
            elif j == labels.shape[1] - 1:
                class0 = max(gr[2 * i + 1, 2 * j, 0], gr[2 * i + 1, 2 * j, 1], gr[2 * i - 1, 2 * j, 0],
                             gr[2 * i - 1, 2 * j, 2], gr[2 * i, 2 * j - 1, 0], gr[2 * i, 2 * j - 1, 2])
                class1 = max(gr[2 * i + 1, 2 * j, 2], gr[2 * i + 1, 2 * j, 3], gr[2 * i - 1, 2 * j, 1],
                             gr[2 * i - 1, 2 * j, 3], gr[2 * i, 2 * j - 1, 1], gr[2 * i, 2 * j - 1, 3])
                if class1 > class0:
                    labels[i, j] = 1
            else:
                class0 = max(gr[2 * i + 1, 2 * j, 0], gr[2 * i + 1, 2 * j, 1], gr[2 * i, 2 * j + 1, 0],
                             gr[2 * i, 2 * j + 1, 1], gr[2 * i - 1, 2 * j, 0], gr[2 * i - 1, 2 * j, 2],
                             gr[2 * i, 2 * j - 1, 0], gr[2 * i, 2 * j - 1, 2])
                class1 = max(gr[2 * i, 2 * j - 1, 3], gr[2 * i - 1, 2 * j, 3], gr[2 * i - 1, 2 * j, 1],
                             gr[2 * i, 2 * j + 1, 3], gr[2 * i, 2 * j + 1, 2], gr[2 * i + 1, 2 * j, 3],
                             gr[2 * i + 1, 2 * j, 2])

                if class1 > class0:
                    labels[i, j] = 1
    return image_is_good, labels


@njit(fastmath=True)
def labeling(gr, epsilon_lower_border):
    print("labeling...")
    epsilon = -np.inf
    for i, j in np.ndindex(gr.shape[:2]):
        if (i + j) % 2 != 0:
            a = gr[i, j].max() - gr[i, j].min()
            if a > epsilon:
                epsilon = a
    print("Start epsilon: ", epsilon)
    labels = np.zeros(((gr.shape[0] + 1) // 2, (gr.shape[1] + 1) // 2), np.uint8)
    for i in range(100):
        gr = deletion_alg(gr, epsilon)
        temp_labeling = labels.copy()
        image_is_good, labels = try_build_labeling(gr, labels)
        if image_is_good and epsilon > epsilon_lower_border:
            epsilon *= 0.5
            print("For epsilon = ", epsilon, ": looking...")
        else:
            print("Final labeling founded;\n Final epsilon = ", epsilon)
            # labels = temp_labeling
            break
    return gr, temp_labeling
