# Watermark removal script

Script used to remove date watermark from videos using color segmentation. Actually only segment orange and blue 
but will be added more colors.

## How to add new segmentation colors? 

To obtain HSV threshold will be used the *range_detector.py* script. These threshold obtained, must be added to
thresholds class(*ColorsHSVThresholds*) in *watermark_removal.py* script.