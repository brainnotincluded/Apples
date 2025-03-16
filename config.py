import os

# Base directory
base_dir = os.path.expanduser("~/PycharmProjects/apples")

# cut_frame settings
path_to_video_cut_frame = os.path.join(base_dir, 'video', 'output_1.avi')
folder_to_save_cut_frame = 'cut_frame'
title_cut = "Cut frame"
cut_frame_version = 1

# --------

# cut_video settings
input_cut_video_path = os.path.join(base_dir, 'video', 'output_0.avi')
output_cut_video_path = os.path.join(base_dir, 'video', 'cut_output_0.mp4')
title_video_cut = "Video_cut"

# --------

# main settings
title_main = "Detected Apples"
output_dir = "output_apples"
path_to_video = os.path.join(base_dir, 'video', 'Example_1.mp4')

# --------

# main pipeline
video_path = "out.mp4"  # Use 0 for webcam, or specify a video file path
window_name = "Apples"
count_frame = 3000
all_the_way = False

# Adjust paths for MacBook
tracker_path = os.path.join(base_dir, 'weights', 'tracker.pt')
classification_model_path = os.path.join(base_dir, 'weights', 'classification.pt')

video_save_path = os.path.join(base_dir, 'save')
img_save_path = os.path.join(base_dir, 'save')
