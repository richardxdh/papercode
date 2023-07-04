#!/usr/bin/env python
# -*- coding: utf-8 -*-
import functools

import numpy as np
import cv2

TT_OTSU = "otsu"
TT_VE = "valley-emphasis"
TT_NVE = "neighborhood-valley-emphasis"
TT_GVE = "gaussian-valley-emphasis"
TT_LVE = "lda-valley-emphasis"
TT_WOV = "weighted-object-variance"
TT_EVE = "entropy-valley-emphasis"
TT_CVE = "cao-valley-emphasis"
TT_DVE = "derivative-valley-emphasis" # derivative-based valley emphasis, namely DVE


def decorator_otsu_based_threshold(threshold_type=None, dtype=np.float64):

    def _decorator_otsu_based_threshold(threshold_func):
        @functools.wraps(threshold_func)
        def wrapper_otsu_based_threshold(gray, mask, **kwargs):

            hist = cv2.calcHist([gray], [0], mask, [256], [0, 256], accumulate=False).flatten().astype(dtype)
            X = np.arange(0, 256)

            if threshold_type == TT_DVE:
                k = kwargs.get("k")
                hist = cv2.blur(hist, (1, 2*k+1)).flatten()

            p = hist / hist.sum()
            mu = np.sum(X * p)

            kwargs["hist"] = hist
            kwargs["X"] = X
            kwargs["p"] = p
            kwargs["mu"] = mu
            if threshold_type == TT_EVE:
                positive_p = p[np.argwhere(p > 0)]
                Hn = -np.sum(positive_p * np.log(positive_p))
                kwargs.update({"Hn": Hn, "Ps": 0, "Hs": 0})

            elif threshold_type == TT_DVE:
                derivative_2nd_p = [p[max(0, i-1)] - 2*p[i] + p[min(255, i+1)] for i in range(256)]
                # derivative_1st_p = [p[min(255, i+1)] - p[i] for i in range(256)]
                # derivative_2nd_p = [derivative_1st_p[min(255, i+1)] - derivative_1st_p[i] for i in range(256)]
                kwargs["derivative_2nd_p"] = derivative_2nd_p
                kwargs["min_derivative_2nd_p"] = np.min(derivative_2nd_p)
                kwargs["max_derivative_2nd_p"] = np.max(derivative_2nd_p)

            max_var= 0
            threshold = 0
            omega_1 = 0
            mu_k = 0

            for t in X:
                omega_1 = omega_1 + p[t]
                omega_2 = 1 - omega_1
                mu_k = mu_k + t * p[t]
                mu_1 = mu_k / omega_1 if omega_1 != 0 else np.nan
                mu_2 = (mu - mu_k) / omega_2 if omega_2 != 0 else np.nan

                kwargs["t"] = t
                var, kwargs = threshold_func(omega_1, mu_1, omega_2, mu_2, **kwargs)

                if var > max_var:
                    max_var = var
                    threshold = t

            return threshold
        return wrapper_otsu_based_threshold
    return _decorator_otsu_based_threshold


@decorator_otsu_based_threshold()
def otsu(omega_1, mu_1, omega_2, mu_2, **kwargs):
    """
    An implementation of
    N. Otsu. A threshold selection method from gray-level histogram.
    IEEE Transactions on Systems, Man and Cybernetics, 9(1):62-66, 1979.
    """
    var = omega_1*(mu_1**2) + omega_2*(mu_2**2)
    return var, kwargs


@decorator_otsu_based_threshold()
def valley_emphasis(omega_1, mu_1, omega_2, mu_2, **kwargs):
    """
    An implementation of
    H. F. Ng. Automatic thresholding for defect detection.
    Pattern Recognition Letters, 27(14):1644-1649, 2006.
    """
    p = kwargs.get("p")
    t = kwargs.get("t")
    var = (1 - p[t]) * (omega_1 * (mu_1 ** 2) + omega_2 * (mu_2 ** 2))
    return var, kwargs


@decorator_otsu_based_threshold()
def neighbor_valley_emphasis(omega_1, mu_1, omega_2, mu_2, **kwargs):
    """
    An implementation of
    J. L. Fan and B. Lei. A modified valley-emphasis method for automatic thresholding.
    Pattern Recognition Letters, 33(6):703-708, 2012.
    """
    p = kwargs.get("p")
    t = kwargs.get("t")
    N = kwargs.get("N")
    sum_of_neighbors = sum(p[max(0, t - N):min(256, t + N + 1)])
    var = (1 - sum_of_neighbors) * (omega_1 * (mu_1 ** 2) + omega_2 * (mu_2 ** 2))
    return var, kwargs


@decorator_otsu_based_threshold()
def gaussian_valley_emphasis(omega_1, mu_1, omega_2, mu_2, **kwargs):
    """
    An implementation of
    H. F. Ng, D. Jargalsaikhan, H. C. Tsai, and C. Y. Lin.
    An improved method for image thresholding based on the valley-emphasis method.
    In Signal and Information Processing Association Annual Summit and Conference (APSIPA),
    2013 Asia-Pacific, pages 1-4, Kaohsiung, Taiwan, 2013. IEEE.
    """
    p = kwargs.get("p")
    t = kwargs.get("t")
    X = kwargs.get("X")
    sigma = kwargs.get("sigma")

    W_t = 0
    for x in X:
        W_t = W_t + p[x] * np.exp(-(x - t) ** 2 / (2 * (sigma ** 2)))
    var = (1 - W_t) * (omega_1 * (mu_1 ** 2) + omega_2 * (mu_2 ** 2))
    return var, kwargs


@decorator_otsu_based_threshold(dtype=np.float32)
def lda_valley_emphasis(omega_1, mu_1, omega_2, mu_2, **kwargs):
    """
    An implementation of
    Z. Liu, J. Wang, Q. Zhao, and C. Li. A fabric defect detection algorithm based on improved valley-emphasis method.
    Research Journal of Applied Sciences, Engineering and Technology, 7(12):2427-2431, 2014.
    """
    p = kwargs.get("p")
    t = kwargs.get("t")
    hist = kwargs.get("hist")

    m1 = (1 - p[t]) * (omega_1 * (mu_1 ** 2) + omega_2 * (mu_2 ** 2))
    S_1 = 0
    S_2 = 0
    for j in range(t + 1):
        S_1 = S_1 + (hist[j] - mu_1) ** 2
    for j in range(t + 1, 256):
        S_2 = S_2 + (hist[j] - mu_2) ** 2
    m2 = S_1 + S_2
    var = m1 / m2
    return var, kwargs


@decorator_otsu_based_threshold()
def wov_valley_emphasis(omega_1, mu_1, omega_2, mu_2, **kwargs):
    """
    An implementation of
    Xiao-cui Yuan, Lu-shen Wu, Qingjin Peng,
    An improved Otsu method using the weighted object variance for defect detection,
    Applied Surface Science 349 (2015) 472–484, 2015
    """
    var = omega_1 * omega_1 * (mu_1 ** 2) + omega_2 * (mu_2 ** 2)
    return var, kwargs


@decorator_otsu_based_threshold(threshold_type=TT_EVE)
def entropy_otsu(omega_1, mu_1, omega_2, mu_2, **kwargs):
    """
    An implementation of
    M. T. N. Truong and S. Kim. Automatic image thresholding using Otsu’s method and entropy weighting scheme for surface defect detection
    Journal of Soft Computing 2017
    """
    p = kwargs.get("p")
    t = kwargs.get("t")
    Hn = kwargs.get("Hn")
    Hs = kwargs.get("Hs")
    Ps = kwargs.get("Ps")

    Ps = Ps + p[t]
    kwargs["Ps"] = Ps
    if p[t] != 0:
        Hs = Hs - p[t] * np.log(p[t])
        kwargs["Hs"] = Hs
    psi = np.log(Ps * (1 - Ps)) + Hs / Ps + (Hn - Hs) / (1 - Ps) if Ps > 0 and Ps < 1 else np.nan
    var = psi * (omega_1 * (mu_1 ** 2) + omega_2 * (mu_2 ** 2))
    return var, kwargs


@decorator_otsu_based_threshold()
def cao_valley_emphasis(omega_1, mu_1, omega_2, mu_2, **kwargs):
    """
    An implement of
    Cao, Xinhua & Li, Taihao & Li, Hongli & Xia, Shunren & Ren, Fuji & Sun, Ye & Xu, Xiaoyin. (2018).
    A Robust Parameter-Free Thresholding Method for Image Segmentation.
    IEEE Access. PP. 1-1. 10.1109/ACCESS.2018.2889013. 2018
    """
    mu = kwargs.get("mu")
    var = omega_1 * omega_2 * ((mu_1 - mu_2)**2 + (mu_1 - mu)**2 + (mu_2 - mu)**2)
    return var, kwargs


@decorator_otsu_based_threshold(threshold_type=TT_DVE)
def derivative_valley_emphasis(omega_1, mu_1, omega_2, mu_2, **kwargs):
    """
    An implement of
    Xing, Jiangwa & Yang, Pei & Qingge, Letu. (2020).
    Automatic thresholding using modified valley emphasis.
    IET Image Processing. 14. 10.1049/iet-ipr.2019.0176.
    2019
    """
    mu = kwargs.get("mu")
    derivative_2nd_p = kwargs.get("derivative_2nd_p")
    min_derivative_2nd_p = kwargs.get("min_derivative_2nd_p")
    max_derivative_2nd_p = kwargs.get("max_derivative_2nd_p")
    t = kwargs.get("t")

    cur_derivative_2nd_p = derivative_2nd_p[t]
    Wv = (cur_derivative_2nd_p - min_derivative_2nd_p) / (max_derivative_2nd_p - min_derivative_2nd_p)
    var = (omega_1 * ((mu_1 - mu)**2) + omega_2 * ((mu_2 - mu)**2)) * Wv
    return var, kwargs

