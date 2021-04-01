#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2      as cv
import numpy    as np
import argparse
import os

from operator import xor
from signal   import signal, SIGINT

'''
Class used to grab images and remove watermark from it.
Usage:
  python removal.py --webcam
  python removal.py --video <path_to_video>
  python removal.py --video <path_to_video> --save
Keys:
  q, Q, Esc - exit
'''

def get_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument('-v', '--video', required=False,
                    help='Path to the video')
    ap.add_argument('-w', '--webcam', required=False,
                    help='Use webcam', action='store_true')

    ap.add_argument('-s', '--save', required=False,
                    help='Save result video', action='store_true')
    args = vars(ap.parse_args())

    if not xor(bool(args['video']), bool(args['webcam'])):
        ap.error('Please specify only one image source')
    return args


class ColorsHSVThresholds(object):
    def __init__(self):
        # format: [(h_min, s_min, v_min), (h_max, s_max, v_max)] 
        self.__hsv_orange_thres = ((0  , 171, 211), (32 , 255, 255))
        self.__hsv_blue_thres   = ((100, 100, 100), (255, 255, 255)) 
    
        self.__hsv_thres_list   = (self.__hsv_orange_thres, self.__hsv_blue_thres)

    def get_threshold_list(self):
        return self.__hsv_thres_list

    def get_selector_label(self):
        selector = '0 : Orange \n1 : Blue'
        return selector


class WatermarkRemover(object):
    def __init__(self, args):
        input_vid  = args['video'] if (not args['webcam']) else 0
        self.__cap = cv.VideoCapture(input_vid)

        if (not self.__cap.isOpened()):
            raise Exception('Could not open video device')
        w, h, fps = self.__get_capture_info(input_vid)

        self.__video_output  = None
        if(args['save'] and not args['webcam']):
            # NOTE. There is a bug when try to define the codecs to save video in mp4, that's why video output format.
            # fourcc = cv.VideoWriter_fourcc(*'XVID') if (splitted_video_path[1] == '.avi') else cv.VideoWriter_fourcc('M','J','P','G')
            splitted_video_path = os.path.splitext(args['video'])
            fourcc = cv.VideoWriter_fourcc(*'XVID')
            self.__video_output = cv.VideoWriter('%s_without_watermark%s' %(splitted_video_path[0], '.avi'), fourcc, fps, (w, h))
            if not self.__video_output.isOpened():
                raise Exception('Could not create output video')

        colors_thres            = ColorsHSVThresholds()
        self.__hsv_thres_list   = colors_thres.get_threshold_list()
        self.__used_hsv_thres   = self.__hsv_thres_list[0]

        self.__window_name = 'Original vs. Segmented vs. Result'
        cv.namedWindow(self.__window_name, cv.WINDOW_AUTOSIZE)
        cv.createTrackbar(colors_thres.get_selector_label(), self.__window_name, 0, len(self.__hsv_thres_list), self.__print_cb)

        self.__dilate_kernel_size = (5, 5)

    def __del__(self):
        self.__cap.release()
        if (self.__video_output):
            self.__video_output.release()

        cv.destroyAllWindows()
        print('Finished conversion.')

    def __get_capture_info(self, input_vid):
        width  = int(self.__cap.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(self.__cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        fps    = self.__cap.get(cv.CAP_PROP_FPS)
        print('Input video info: %s -> [width: %i, height: %i, fps: %i]' %(input_vid, width, height, fps))
        return width, height, fps

    def __print_cb(self, color_idx):
        self.__used_hsv_thres = self.__hsv_thres_list[color_idx]
        pass

    def __rescale_image(self, frame, scale_percent = 40):
        width   = int(frame.shape[1] * scale_percent / 100)
        height  = int(frame.shape[0] * scale_percent / 100)
        dim     = (width, height)
        resized = cv.resize(frame, dim, interpolation = cv.INTER_AREA)
        return resized

    def __obtain_and_apply_mask(self, frame):
        hsv         = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        mask_binary = cv.inRange(hsv, self.__used_hsv_thres[0], self.__used_hsv_thres[1])

        kernel       = np.ones(self.__dilate_kernel_size, np.uint8)
        mask_binary  = cv.dilate(mask_binary, kernel, iterations = 1)
        mask_segment = cv.bitwise_and(frame, frame, mask = mask_binary)  
        return mask_segment, mask_binary

    def __apply_inpaint(self, frame, mask):
        output = cv.inpaint(frame, mask, 3, cv.INPAINT_TELEA)
        return output

    def __save_image(self, frame, name, img_format = '.jpg'):
        cv.imwrite(name + img_format, frame)

    def __get_blank_image(self, shape):
        img = np.zeros((shape), np.uint8)
        return img

    def __show_images(self, original_img, segmented_img, final_img):
        h_imgs_1 = cv.hconcat([original_img, segmented_img])
        h_imgs_2 = cv.hconcat([final_img, self.__get_blank_image(final_img.shape)])
        v_imgs   = cv.vconcat([h_imgs_1, h_imgs_2])
        cv.imshow(self.__window_name, v_imgs)

    def run(self):
        while(True):
            ret, frame = self.__cap.read()
            if (not ret):
                break

            resized_frame         = frame.copy()
            resized_frame         = self.__rescale_image(frame.copy())
            segmented_frame, mask = self.__obtain_and_apply_mask(resized_frame)
            result_frame          = self.__apply_inpaint(resized_frame, mask)

            self.__show_images(resized_frame, segmented_frame, result_frame)
            
            if (self.__video_output):
                real_size_frame = frame.copy()
                real_size_segmented_frame, real_size_mask = self.__obtain_and_apply_mask(real_size_frame)
                real_size_result_frame                    = self.__apply_inpaint(real_size_frame, real_size_mask)
                self.__video_output.write(real_size_result_frame)

            key = cv.waitKey(1) & 0xFF
            if ((key == ord('q')) or (key == ord('Q')) or (key == 27)):
                break
        self.__del__()

    def sig_handler(self, signal_received, frame):
        print('SIGINT or CTRL-C detected. Exiting gracefully')
        self.__del__()

if __name__ == '__main__':
    remover = WatermarkRemover(get_arguments())
    signal(SIGINT, remover.sig_handler)
    remover.run()
