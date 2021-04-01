# Watermark removal script

Script used to remove date watermark from videos using color segmentation. Actually only segment orange and blue 
but will be added more colors.

## How to add new segmentation colors? 

To obtain HSV threshold will be used the *range_detector.py* script. These threshold obtained, must be added to
thresholds class([*ColorsHSVThresholds*](https://github.com/mgrova/fpv_video_tools/blob/bd58d06cf2f178d8a989da67c6d7b8647139b85b/watermark_removal/watermark_removal.py#L42)) 
in *watermark_removal.py* script and add the new option to [label selector](https://github.com/mgrova/fpv_video_tools/blob/master/watermark_removal/watermark_removal.py#L48).
## Demo

![Removal script demo](docs/removal_demo.gif)