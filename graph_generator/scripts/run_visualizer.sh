# python -m utils.graph_visualizer \
#     --jsonl_path /home/wangxingjian/DORO-STVG/graph_generator/scene_graphs.jsonl \
#     --video_path /home/wangxingjian/data/vidstg/video/2560825239.mp4 \
#     --output_path /home/wangxingjian/DORO-STVG/graph_generator/output/2560825239.mp4 \

# python3 utils/hc_stvg_visualizer.py \
#     --ann /home/wangxingjian/data/hc-stvg2/annos/test.json \
#     --video-root /home/wangxingjian/data/hc-stvg2/v2_video \
#     --output-dir ./vis_outputs \
#     --video 50_TM5MPJIq1Is.mkv \
#     --limit 1

# python -m utils.vidstg_visualizer \
#     --ann /home/wangxingjian/data/vidstg/extracted_data.json \
#     --video-root /home/wangxingjian/data/vidstg \
#     --output-dir /home/wangxingjian/DORO-STVG/graph_generator/output/2451862413.mp4 \
#     --ann-id 56677

python3 utils/query_web_viewer.py \
    --jsonl /home/wangxingjian/DORO-STVG/graph_generator/output/query1.jsonl \
    --host 127.0.0.1\
    --port 8765
