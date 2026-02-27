#!/bin/bash
python utils/video_reencode.py \
    --input /home/wangxingjian/data/hc-stvg2/v2_video/50_TM5MPJIq1Is.mkv \
    --output /home/wangxingjian/data/hc-stvg2/v2_video/50_TM5MPJIq1Is_2fps.mp4 \
    --fps 2 \
    --codec libx264  