# -*- coding: utf-8 -*-
"""
An implementation of

Qingming Huang, Wen Gao, Wenjian Cai,
Thresholding technique with adaptive window selection for uneven lighting image,
Pattern Recognition Letters,
Volume 26, Issue 6,
2005,
Pages 801-808,
ISSN 0167-8655,
https://doi.org/10.1016/j.patrec.2004.09.035.
(https://www.sciencedirect.com/science/article/pii/S0167865504002648)
Abstract: By adaptively selecting image window size based on the pyramid data structure manipulation of Lorentz information measure, a new technique for image thresholding is proposed. The advantage of this technique is its effectiveness in eliminating both uneven lighting disturbance and ghost objects. When applied to Otsu’s thresholding approach, it can provide accurate result under uneven lighting disturbance while using Otsu’s method alone cannot. Experimental results show the effectiveness of this method.
Keywords: Image segmentation; Thresholding; Adaptive window selection; Lorentz information measure
"""

import numpy as np
import cv2
from otsu_based_threshold import otsu


def otsu_threshold_1d(imgdata):
    hist = np.histogram(imgdata, bins=range(257))[0].astype(np.float64)
    X = np.arange(0, 256)

    p = hist / hist.sum()
    mu = np.sum(X * p)

    max_var = 0
    threshold = 0
    omega_1 = 0
    mu_k = 0

    for t in X:
        omega_1 = omega_1 + p[t]
        omega_2 = 1 - omega_1
        mu_k = mu_k + t * p[t]
        mu_1 = mu_k / omega_1 if omega_1 != 0 else np.nan
        mu_2 = (mu - mu_k) / omega_2 if omega_2 != 0 else np.nan

        var = omega_1 * (mu_1 ** 2) + omega_2 * (mu_2 ** 2)

        if var > max_var:
            max_var = var
            threshold = t

    return threshold


def lorentz_infomation_measure(gray):

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256], accumulate=False).flatten().astype(np.float64)
    sorted_hist = np.sort(hist, kind="stable")
    sorted_p = sorted_hist / sorted_hist.sum()

    S = [np.sum(sorted_p[:k+1]) for k in range(len(sorted_p))]
    lim = np.sum(S)

    return lim


def build_feature_image(gray, a, b):
    H, W = gray.shape
    m = W // a
    n = H // b
    feature_image = np.zeros((n, m, 5), dtype=np.float64)
    for j in range(n):
        for i in range(m):
            feature_image[j, i] = (
                lorentz_infomation_measure(gray[j*b:(j+1)*b, i*a:(i+1)*a]), # Lorentz Information Measure
                i*a, j*b, a, b # window info: x, y, w, h
            )

    return feature_image.reshape((-1, 5))


def threshold_adaptive_window_selection(gray, a=10, b=10):

    raw_gray = gray.copy()
    rawH, rawW = raw_gray.shape
    # Pad the image to make its length and width integral multiples of a and b respectively.
    if rawW % a > 0:
        W = (rawW//a + 1) * a
        tmp = np.zeros((rawH, W), dtype=np.uint8)
        tmp[:, :rawW] = gray
        tmp[:, rawW:] = np.flip(gray[:, -(W-rawW):], axis=1)
        gray = tmp
    if rawH % b > 0:
        H = (rawH//b + 1) *b
        tmp = np.zeros((H, gray.shape[1]), dtype=np.uint8)
        tmp[:rawH, :] = gray
        tmp[rawH:, :] = np.flip(gray[-(H-rawH):, :], axis=0)
        gray = tmp
    H, W = gray.shape

    processed_mask = np.zeros_like(gray)
    threshold_mask = np.zeros_like(gray)

    # step1/2/3
    features_data = build_feature_image(gray, a, b)
    feature_start = 0

    while True:

        # step4/5
        lim_feature = features_data[:, 0]
        lim_thresh = otsu_threshold_1d(lim_feature)
        print(f"lim threshold: {lim_thresh}, a={a}, b={b}")

        # step6
        greater_lim_index = np.argwhere(lim_feature > lim_thresh).flatten()
        greater_lim_index = greater_lim_index[np.where(greater_lim_index >= feature_start)]
        for idx in greater_lim_index:
            x, y, w, h  = np.int32(features_data[idx, 1:])
            window_mask = np.zeros_like(gray)
            window_mask[y:y+h, x:x+w] = 255
            processing_mask = cv2.subtract(window_mask, processed_mask) # mask of have not been thresholded pixels
            if np.count_nonzero(processing_mask) == 0:
                # print("no pixel in this window needs to be thresholded")
                continue

            window_threshold = otsu(gray, processing_mask)
            window_foremask = cv2.threshold(gray, window_threshold, 255, cv2.THRESH_BINARY)[1]
            window_foremask = cv2.bitwise_and(window_foremask, processing_mask)

            processed_mask = cv2.bitwise_or(processed_mask, processing_mask)
            threshold_mask = cv2.bitwise_or(threshold_mask, window_foremask)

        # step7
        if np.count_nonzero(processed_mask) == processed_mask.size:
            print("the whole image has been thresholded")
            break

        # step8
        if np.count_nonzero(lim_feature[feature_start:] <= lim_thresh) == 0:
            print("there is no pixel in f' with gray level value less than or equal to T'")
            break

        # step9
        a = min(2*a, W)
        b = min(2*b, H)
        if a == W and b == H:
            window_mask = np.full_like(gray, 255)
            processing_mask = cv2.subtract(window_mask, processed_mask)

            T_image = otsu(gray, None)
            window_foremask = cv2.threshold(gray, T_image, 255, cv2.THRESH_BINARY)[1]
            window_foremask = cv2.bitwise_and(window_foremask, processing_mask)

            processed_mask = cv2.bitwise_or(processed_mask, processing_mask)
            threshold_mask = cv2.bitwise_or(threshold_mask, window_foremask)
            break

        less_lim_index = np.argwhere(lim_feature <= lim_thresh).flatten()
        less_lim_index = less_lim_index[np.where(less_lim_index >= feature_start)]
        for idx in less_lim_index:
            x, y = np.int32(features_data[idx, 1:3])
            if x+a > W:
                x = W - a
            if y+b > H:
                y = H - b
            lim = lorentz_infomation_measure(gray[y:y+b, x:x+a])
            feature = np.array([[lim, x, y, a, b]])
            features_data = np.concatenate([features_data, feature], axis=0)

        feature_start = lim_feature.size

    # step10: END here

    final_mask = threshold_mask[:rawH, :rawW]
    return final_mask


if __name__ == "__main__":

    from skimage import data
    from toolkits import show_image

    img = data.coins()
    print(img.shape[::-1])
    threshold_mask = threshold_adaptive_window_selection(img, 96, 60)

    cnts = cv2.findContours(threshold_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    img2 = img.copy()
    cv2.drawContours(img2, cnts, -1, 255, -1)
    show_image(np.concatenate([img, img2], axis=1))

    pass