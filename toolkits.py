# -*- coding: utf8 -*-

import platform
from screeninfo import get_monitors

primary_monitor = [m for m in get_monitors() if m.is_primary][0]
SCR_WIDTH = primary_monitor.width
SCR_HEIGHT = primary_monitor.height

import numpy as np
import cv2

def show_image(img, delay=0, fullscreen=True, title="show image", keymap={"e": "exit", "n": "next"}):
    """
    img:    it will be shown.
    delay:  display time, unit is millisecond
    fullscreen: whether show img in full screen
    """

    if platform.system().lower() == "darwin":
        fullscreen = False

    if len(img.shape) == 2:
        img = np.stack([img]*3, axis=-1)

    # img = show_keymap(img, keymap)
    keys = [k for k in keymap.keys()]
    exit_keys = [k for k in keymap.keys() if keymap[k]=="exit"]

    if fullscreen:
        cv2.namedWindow(title, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    else:
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        h, w = img.shape[:2]
        cv2.resizeWindow(title, w, h)
        x = max(0, (SCR_WIDTH- w) // 2)
        y = max(0, (SCR_HEIGHT- h) // 2)
        cv2.moveWindow(title, x, y)

    cv2.imshow(title, img)
    press_key = None
    while True:
        waitkey = cv2.waitKey(delay)
        if waitkey == -1:
            break

        press_key = chr(waitkey & 0xFF)
        if press_key in keys:
            break

    if press_key in exit_keys:
        cv2.destroyAllWindows()
    return press_key
