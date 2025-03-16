import logging
import cv2
import time

import torch

from configs.config import *
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
from tqdm import tqdm

logging.getLogger('ultralytics').setLevel(logging.WARNING)


def draw_custom_annotations(frame, results, tracker_d, class_info):
    output = []
    boxes = results.boxes
    if len(boxes) == 0:
        return tracker_d, output

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    for obj_idx, obj in enumerate(boxes):
        x1, y1, x2, y2 = map(int, obj.xyxy[0].tolist())
        D = max(abs(x2 - x1), abs(y2 - y1))
        track_id = obj.id.item() if obj.id is not None else None

        if track_id is None:
            continue

        if obj_idx < len(class_info):
            class_label, prob = class_info[obj_idx]
        else:
            class_label, prob = 'unknown', 0.0

        if track_id not in tracker_d:
            tracker_d[track_id] = {
                'general': {'count': 0, 'sum_bad': 0, 'sum_D': 0, 'sum_median_h': 0},
                'history': []
            }

        h_channel = hsv_frame[y1:y2, x1:x2, 0]
        median_h = np.median(h_channel)

        tracker_d[track_id]['history'].append({
            'label': class_label,
            'prob': prob,
            'D': D,
            'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2,
            'median_h': median_h
        })

        general = tracker_d[track_id]['general']
        general['count'] += 1
        general['sum_bad'] += int(class_label == 'bad')
        general['sum_D'] += D
        general['sum_median_h'] += median_h

        if y2 < 30:
            general['line'] = 1 if x2 > 700 else 2
            output.append(track_id)

        color = (0, 255, 0) if class_label == "good" else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f'Id {track_id} {class_label} {prob:.2f} D={D:.2f}'
        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return tracker_d, output


def main():
    start = time.time()
    tracker_d = defaultdict(dict)
    video = cv2.VideoCapture(video_path)
    batch_size = 1

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT)) if not all_the_way else count_frame
    progress_bar = tqdm(total=total_frames, desc="Processing Frames", unit="frame")

    while True:
        batch_frames, frames_processed = [], 0
        while len(batch_frames) < batch_size:
            ret, frame = video.read()
            if not ret or (not all_the_way and frames_processed >= count_frame):
                break
            batch_frames.append(frame)
            frames_processed += 1

        if not batch_frames:
            break

        batch_results = tracker_model.track(batch_frames, persist=True, conf=0.3, verbose=False)
        all_crops, frame_indices, obj_indices = [], [], []

        for i, res in enumerate(batch_results):
            for j, box in enumerate(res.boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cropped = batch_frames[i][y1:y2, x1:x2]
                all_crops.append(cropped)
                frame_indices.append(i)
                obj_indices.append(j)

        class_info = [[] for _ in batch_frames]
        if all_crops:
            classifications = classification_model.predict(all_crops, conf=0.25, verbose=False, batch=8)
            for k, res in enumerate(classifications):
                i, j = frame_indices[k], obj_indices[k]
                prob = res.probs.cpu().data.numpy()
                class_idx = np.argmax(prob)
                class_info[i].append((res.names[class_idx], prob[class_idx]))

        for i, frame in enumerate(batch_frames):
            res = batch_results[i]
            current_class_info = class_info[i] if i < len(class_info) else []
            tracker_d, _ = draw_custom_annotations(frame, res, tracker_d, current_class_info)
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        progress_bar.update(len(batch_frames))
        if not all_the_way and progress_bar.n >= count_frame:
            break

    progress_bar.close()
    video.release()
    cv2.destroyAllWindows()

    total_bad = sum(v['general']['sum_bad'] for v in tracker_d.values())
    print(f"\nTotal Bad Apples: {total_bad}")
    print(f"Avg processing time: {(time.time() - start) / progress_bar.n:.2f}s/frame")


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'mps'
    print(f"Using device: {device}")
    tracker_model = YOLO(tracker_path).to(device)
    classification_model = YOLO(classification_model_path).to(device)
    main()