import logging
import time
from collections import defaultdict
from queue import Queue
from threading import Thread

import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO

from configs.config import *

logging.getLogger('ultralytics').setLevel(logging.WARNING)

RESIZE_SIZE = (320, 320)
SKIP_FRAMES = 4
MAX_TRACKS = 16
GRAPH_CONFIG = {'width': 200, 'height': 80, 'margin': 20, 'max_fps': 60, 'history_len': 100}


def draw_fps_graph(frame, fps, history):
    bg_color = (40, 40, 40)
    text_color = (0, 255, 0)
    graph_base = (GRAPH_CONFIG['margin'], GRAPH_CONFIG['margin'])

    cv2.rectangle(frame, graph_base, (graph_base[0] + GRAPH_CONFIG['width'], graph_base[1] + GRAPH_CONFIG['height']),
                  bg_color, -1)

    if len(history) > 1:
        for i in range(1, len(history)):
            x1 = graph_base[0] + int((i - 1) * GRAPH_CONFIG['width'] / (len(history) - 1))
            y1 = graph_base[1] + GRAPH_CONFIG['height'] - int(
                history[i - 1] / GRAPH_CONFIG['max_fps'] * GRAPH_CONFIG['height'])
            x2 = graph_base[0] + int(i * GRAPH_CONFIG['width'] / (len(history) - 1))
            y2 = graph_base[1] + GRAPH_CONFIG['height'] - int(
                history[i] / GRAPH_CONFIG['max_fps'] * GRAPH_CONFIG['height'])
            cv2.line(frame, (x1, y1), (x2, y2), text_color, 1)

    cv2.putText(frame, f'FPS: {fps:.1f}', (graph_base[0] + 10, graph_base[1] + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                text_color, 2)


class AsyncVideoLoader:
    def __init__(self, path, queue_size=128):
        self.cap = cv2.VideoCapture(path)
        self.queue = Queue(maxsize=queue_size)
        self.stopped = False
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def start(self):
        Thread(target=self.update).start()
        return self

    def update(self):
        while not self.stopped:
            if not self.queue.full():
                ret, frame = self.cap.read()
                if not ret:
                    self.stopped = True
                else:
                    self.queue.put((ret, frame))
            else:
                time.sleep(0.01)

    def read(self):
        return self.queue.get()

    def stop(self):
        self.stopped = True
        self.cap.release()


def main():
    device = 'cuda' if torch.cuda.is_available() else 'mps'

    tracker = YOLO(tracker_path)
    tracker.fuse()
    tracker.model.to(device).half()

    classifier = YOLO(classification_model_path)
    classifier.fuse()
    classifier.model.to(device).half()

    video_loader = AsyncVideoLoader(video_path).start()
    progress = tqdm(total=video_loader.total_frames, desc="Processing", unit="frame",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]")

    tracking_data = defaultdict(lambda: {'frame_count': 0, 'class_history': [], 'last_position': (0, 0)})

    fps_history = []
    start_time = time.time()
    processed_frame = None
    while True:
        ret, frame = video_loader.read()
        if not ret:
            break

        current_fps = (progress.n + 1) / (time.time() - start_time)
        fps_history.append(current_fps)
        fps_history = fps_history[-GRAPH_CONFIG['history_len']:]
        progress.set_postfix_str(f"FPS: {current_fps:.1f}")

        display_frame = frame.copy()

        if progress.n % SKIP_FRAMES == 0:
            with torch.no_grad():
                results = tracker.track(frame, persist=True, conf=0.3, max_det=MAX_TRACKS, verbose=False)

            if results and results[0].boxes:
                frame_results = results[0]
                boxes = frame_results.boxes.cpu().numpy()

                for idx, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    track_id = box.id.item()

                    tracking_data[track_id]['frame_count'] += 1
                    frame_count = tracking_data[track_id]['frame_count']

                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue

                    resized = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), RESIZE_SIZE)
                    tensor = torch.from_numpy(resized).permute(2, 0, 1).to(device).half() / 255

                    with torch.no_grad():
                        cls_result = classifier(tensor[None])[0]

                    cls = classifier.names[cls_result.probs.top1] if cls_result.probs else 'unknown'
                    prob = cls_result.probs.top1conf.item() if cls_result.probs else 0.0

                    tracking_data[track_id]['class_history'].append((cls, prob))
                    tracking_data[track_id]['last_position'] = (x1, y1)

                    color = (0, 255, 0) if cls == "good" else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    text = f'ID {track_id} {cls} {prob:.2f} Frames: {frame_count}'
                    cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                processed_frame = frame.copy()

        if processed_frame is not None:
            display_frame = processed_frame.copy()

        draw_fps_graph(display_frame, current_fps, fps_history)
        cv2.imshow('Apple Tracker', display_frame)
        progress.update(1)

        if cv2.waitKey(1) & 0xFF == ord('q') or progress.n == video_loader.total_frames:
            break
    print("\nTracking Statistics:")

    sorted_apples = sorted(tracking_data.items(), key=lambda x: x[1]['frame_count'], reverse=True)

    for track_id, data in sorted_apples:
        print(f"\nApple {track_id}:")
        print(f" - Frames tracked: {data['frame_count']}")
        print(f" - Last position: {data['last_position']}")
        print(f" - Most recent classification: {data['class_history'][-1][0] if data['class_history'] else 'N/A'}")
        print(f" - Average confidence: {np.mean([c[1] for c in data['class_history']]):.2f}")

    print(f"\nFinal Performance: {progress.n / (time.time() - start_time):.1f} FPS")
    video_loader.stop()
    progress.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
