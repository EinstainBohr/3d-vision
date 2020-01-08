#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2

def run(imgdata):
    cv2.imshow("img", imgdata)
    cv2.waitKey(3)
