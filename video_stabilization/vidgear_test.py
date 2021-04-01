#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# import required libraries
from vidgear.gears import VideoGear
from vidgear.gears.stabilizer import Stabilizer
import numpy as np
import cv2   as cv
import sys


'''
Script to test VidGear stabilization.
Usage:
  python vidgear_test.py  <path_to_video>
Keys:
  q - exit
'''
def main_test():
    if (len(sys.argv) > 1):
        video_path = sys.argv[1]
    else:
        raise Exception("It's necessary to pass target video to stabilize")

    # open any valid video stream with stabilization enabled(`stabilize = True`)
    # stream_stab = VideoGear(source=video_path, stabilize=True).start()
    stab = Stabilizer(smoothing_radius=30, crop_n_zoom=True, border_size=1, logging=True)

    # open same stream without stabilization for comparison
    stream_org = VideoGear(source=video_path).start()

    # loop over
    while True:
        # read un-stabilized frame
        scale_percent = 40
        frame_org = stream_org.read()
        width   = int(frame_org.shape[1] * scale_percent / 100)
        height  = int(frame_org.shape[0] * scale_percent / 100)
        dim     = (width, height)
        frame_org_resized = cv.resize(frame_org, dim, interpolation = cv.INTER_AREA)
    
        # read stabilized frames
        # frame_stab = stream_stab.read()
        frame_stab = stab.stabilize(frame_org)

        # check for stabilized frame if Nonetype
        if frame_stab is None:
            break

        width   = int(frame_stab.shape[1] * scale_percent / 100)
        height  = int(frame_stab.shape[0] * scale_percent / 100)
        dim     = (width, height)
        frame_stab_resized = cv.resize(frame_stab, dim, interpolation = cv.INTER_AREA)

        # concatenate both frames
        output_frame = np.concatenate((frame_org_resized, frame_stab_resized), axis=1)

        # put text over concatenated frame
        cv.putText(
            output_frame,
            "Before",
            (10, output_frame.shape[0] - 10),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        cv.putText(
            output_frame,
            "After",
            (output_frame.shape[1] // 2 + 10, output_frame.shape[0] - 10),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        # Show output window
        cv.imshow("Stabilized Frame", output_frame)

        key = cv.waitKey(1) & 0xFF
        if ((key == ord('q')) or (key == ord('Q')) or (key == 27)):
            break

    # close output window
    cv.destroyAllWindows()

    # safely close both video streams
    stream_org.stop()
    # stream_stab.stop()

if __name__ == '__main__':
    main_test()
