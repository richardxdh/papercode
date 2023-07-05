# -*- coding: utf-8 -*-
"""
An implementation of

Pattnaik, Tapaswini & Kanungo, Priyadarshi. (2021).
Adaptive Window Selection for Non-uniform Lighting Image Thresholding.
ELCVIA Electronic Letters on Computer Vision and Image Analysis. 20. 10.5565/rev/elcvia.1301.
"""

import numpy as np
import cv2
from otsu_based_threshold import cao_valley_emphasis as otsu


def evaluate_CI(imgdata, dmu_thresh=0.5, sigma_thresh=60):
    hist = np.histogram(imgdata, bins=range(257))[0].astype(np.float64)
    X = np.arange(0, 256)

    p = hist / hist.sum()
    mu = np.sum(X * p)

    otsu_T = 0
    otsu_mu1 = 0
    otsu_mu2 = 0

    max_var = 0
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
            otsu_T = t
            otsu_mu1 = mu_1
            otsu_mu2 = mu_2

    min_gray_level = np.min(imgdata)
    max_gray_level = np.max(imgdata)
    d_mu = np.abs(otsu_mu1 - otsu_mu2) / (max_gray_level - min_gray_level)

    sigma = np.sqrt(np.sum((X - mu)**2 * p))
    print(d_mu, sigma, otsu_T)

    return d_mu > dmu_thresh and sigma < sigma_thresh


def evaluate_CI2(imgdata, dmu_thresh=0.5, sigma_thresh=60):
    otsu_T = int(cv2.threshold(imgdata, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)[0])

    hist = np.histogram(imgdata, bins=range(257))[0].astype(np.float64)
    X = np.arange(0, 256)

    p = hist / hist.sum()
    mu = np.sum(X * p)

    otsu_mu1 = np.sum(X[:otsu_T+1] * hist[:otsu_T+1] / np.sum(hist[:otsu_T+1]))
    otsu_mu2 = np.sum(X[otsu_T+1:] * hist[otsu_T+1:] / np.sum(hist[otsu_T+1:]))

    min_gray_level = np.min(imgdata)
    max_gray_level = np.max(imgdata)
    d_mu = np.abs(otsu_mu1 - otsu_mu2) / (max_gray_level - min_gray_level)

    sigma = np.sqrt(np.sum((X - mu)**2 * p))
    print(d_mu, sigma, otsu_T)

    return d_mu > dmu_thresh and sigma < sigma_thresh


def threshold_stages(img, mask, thresh_mask, winx, winy, winw, winh, minw, minh):
    # four stages

    unprocessed_win_pos = []

    win_gray = img[winy:winy+winh, winx:winx+winw]
    ci = evaluate_CI2(win_gray)
    if ci:
        otsu_T = otsu(win_gray, None)
        mask[winy:winy+winh, winx:winx+winw] = cv2.threshold(win_gray, otsu_T, 255, cv2.THRESH_BINARY)[1]
        thresh_mask[winy:winy+winh, winx:winx+winw] = otsu_T

    elif  winw <= minw or winh <= minh:
        unprocessed_win_pos.append([winx, winy, winw, winh])

    else:
        sub_winw = winw//2
        sub_winh = winh//2
        for sub_winy in range(winy, winy+winh, sub_winh):
            for sub_winx in range(winx, winx+winw, sub_winw):

                sub_unprocessed_win_pos = \
                    threshold_stages(img, mask, thresh_mask, sub_winx, sub_winy, sub_winw, sub_winh, minw, minh)
                unprocessed_win_pos.extend(sub_unprocessed_win_pos)

    return unprocessed_win_pos


def fine_unprocessed_win(gray, mask, thresh_mask, unprocessed_win_pos):
    #  The minimum size of subimages in stage-IV is 32 × 32 and the maximum size may be 256 × 256. If the minimum size of
    # subimages(32 × 32) resolution at stage-IV does not satisfy CI = 1 then the threshold value of that subimage is
    # evaluated based on the average of the neighbouring subimages.

    unprocessed_mask = np.zeros_like(mask)
    neighborhood_mask = np.zeros_like(mask)
    for pos in unprocessed_win_pos:
        x, y, w, h = pos
        unprocessed_mask[:, :] = 0
        unprocessed_mask[y:y+h, x:x+w] = 1

        x1 = max(0, x-1)
        y1 = max(0, y-1)
        x2 = min(W, x+w+1)
        y2 = min(H, y+h+1)
        neighborhood_mask[:, :] = 0
        neighborhood_mask[y1:y2, x1:x2] = 1
        neighborhood_mask = cv2.subtract(neighborhood_mask, unprocessed_mask)

        neighbor_threshold = neighborhood_mask * thresh_mask
        if np.count_nonzero(neighbor_threshold) == 0:
            continue
        avg_t = np.sum(neighbor_threshold) / np.count_nonzero(neighbor_threshold)

        win_img = gray[y:y+h, x:x+w]
        win_mask = cv2.threshold(win_img, avg_t, 255, cv2.THRESH_BINARY)[1]
        mask[y:y+h, x:x+w] = win_mask

    return mask


def threshold_adawinsel_nmdm(gray, minw=32, minh=32):
    H, W = gray.shape
    mask = np.zeros_like(gray)
    thresh_mask = np.zeros_like(gray)

    unprocessed_win_pos = threshold_stages(gray, mask, thresh_mask, 0, 0, W, H, minw, minh)
    mask = fine_unprocessed_win(gray, mask, thresh_mask, unprocessed_win_pos)

    return mask


if __name__ == "__main__":


    from skimage import data
    from toolkits import show_image

    img = data.coins()
    print(img.shape[::-1])
    H, W = img.shape

    img2 = cv2.resize(img, (256, 256))
    threshold_mask = threshold_adawinsel_nmdm(img2)
    threshold_mask = cv2.resize(threshold_mask, (W, H))

    cnts = cv2.findContours(threshold_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    img2 = img.copy()
    cv2.drawContours(img2, cnts, -1, 255, -1)
    show_image(np.concatenate([img, img2], axis=1))

    pass
