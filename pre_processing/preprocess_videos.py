from VideoProcessor import VideoProcessor
import cv2
import pandas as pd
import os
import torch
import csv


# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Paths
cdir = os.getcwd()
path_video  = os.path.abspath(os.path.join(cdir, "../", "DATA"))
path_trials = os.path.abspath(os.path.join(cdir, "../", "DATA", "metadata"))

# Load model
processor = VideoProcessor(os.path.abspath(os.path.join(cdir,"pre_processing","yolov8n-face-lindevs.pt")), device=device)

# Load trial list
trials = pd.read_csv(
    os.path.abspath(os.path.join(path_trials, "trials_example.csv")),
    sep=",", header=0, na_values="-", dtype=str
)

# Initialize log file
# Stores video metadata (frame count, fps, resolution) for each processed file.
log_path = os.path.abspath(os.path.join(path_trials, "video_frames_log_example.csv"))
with open(log_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["subject_id", "trial_id", "side",
                     "n_frames", "fps", "width", "height"])

# Main loop
for _, trial in trials.iterrows():
    for side in ["left", "right"]:

        input_path  = os.path.abspath(os.path.join(
            path_video, f"{trial.subject_id}_{trial.trial_id}_{side}.MOV"))
        output_path = os.path.abspath(os.path.join(
            path_video, f"{trial.subject_id}_{trial.trial_id}_{side}.mp4"))

        if not os.path.exists(input_path):
            print(f"WARNING: file not found: {input_path}")
            continue

        # Read video metadata before processing
        cap = cv2.VideoCapture(input_path)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps      = cap.get(cv2.CAP_PROP_FPS)
        width    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # Append video metadata to log
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([trial.subject_id, trial.trial_id, side,
                             n_frames, fps, width, height])

        print(f"Processing: {trial.subject_id}_{trial.trial_id}_{side} "
              f"({n_frames} frames)")

        # Anonymize faces
        processor.blur_faces(input_path, output_path)