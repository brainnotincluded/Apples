import logging
import cv2
import time
from config import *
from ultralytics import YOLO
import numpy as np
import os
import torch
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import threading
import uvicorn
import json

# Проверка доступности GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# Отключение сообщений YOLO
logging.getLogger('ultralytics').setLevel(logging.WARNING)

app = FastAPI()
roi_areas = []

def load_config():
    try:
        with open("config.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"roi_areas": []}

def save_config(config):
    with open("config.json", "w") as f:
        json.dump(config, f, indent=4)

def draw_custom_annotations(frame, results, tracker_d, save_dir, frame_count):
    output = []
    for obj in results[0].boxes:
        x1, y1, x2, y2 = map(int, obj.xyxy[0].tolist())
        track_obj = obj.id
        probs = obj.conf.item()  # Преобразуем тензор в число

        if track_obj is not None:
            track_id = track_obj.item()
            cropped_object = frame[y1:y2, x1:x2]
            custom_text = f'{int(track_id)} {round(probs * 100)}%'

            path_to_save = f'{save_dir}/apple_{track_id}'
            if track_id not in tracker_d:
                tracker_d[track_id] = {
                    'general': {
                        'count': 0,
                        'sum_bad': 0,
                        'sum_D': 0,
                        'sum_median_h': 0,
                    },
                    'history': []
                }

            hsv_frame = cv2.cvtColor(cropped_object, cv2.COLOR_BGR2HSV)
            h_channel, s_channel, v_channel = cv2.split(hsv_frame)
            median_h = np.median(h_channel)

            D = ((x2 - x1) + (y2 - y1)) / 2
            tracker_d[track_id]['history'].append({
                'prob': probs,
                'x1': x1,
                'x2': x2,
                'y1': y1,
                'y2': y2,
                'median_h': median_h,
                'D': D
            })
            tracker_d[track_id]['general']['count'] += 1
            tracker_d[track_id]['general']['sum_median_h'] += median_h
            tracker_d[track_id]['general']['sum_D'] += D
            if probs < 0.5:
                tracker_d[track_id]['general']['sum_bad'] += 1

            for i, (rx1, ry1, rx2, ry2) in enumerate(roi_areas):
                if rx1 <= x1 <= rx2 and ry1 <= y1 <= ry2:
                    tracker_d[track_id]['general']['line'] = i + 1
                    output.append(track_id)

            color = (0, 255, 0) if probs > 0.5 else (0, 0, 255)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 2
            text_size, _ = cv2.getTextSize(custom_text, font, font_scale, font_thickness)
            text_x = x1
            text_y = y1 - 10 if y1 - 10 > 10 else y1 + text_size[1] + 10

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, custom_text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)

    return tracker_d, output

@app.get("/cup_number")
async def get_cup_number():
    return JSONResponse(content={"cup_number": 31})

def run_server():
    uvicorn.run(app, host="0.0.0.0", port=8000)

def main():
    global roi_areas

    config = load_config()
    roi_areas = config.get("roi_areas", [])

    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_file = f'Yolo_tracker_gpu.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, 8.0, (frame_width, frame_height))

    os.makedirs(img_save_path, exist_ok=True)

    start = time.time()
    frame_count = 0
    tracker_d = {}

    while all_the_way or frame_count < count_frame:
        if frame_count % 20 == 0:
            print(f'{frame_count}/{count_frame}')
        frame_count += 1
        ret, frame = video.read()
        if not ret:
            break

        for (x1, y1, x2, y2) in roi_areas:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        results = tracker_model.track(frame, persist=True, conf=0.3)
        tracker_d, output = draw_custom_annotations(frame, results, tracker_d, img_save_path, frame_count)

        out.write(frame)
        cv2.imshow('out', frame)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        elif key == ord(" "):
            r = cv2.selectROI("Select ROI", frame)
            roi_areas.append((int(r[0]), int(r[1]), int(r[0] + r[2]), int(r[1] + r[3])))
            config["roi_areas"] = roi_areas
            save_config(config)

        for i in output:
            line = tracker_d[i]['general']['line']
            general_count = tracker_d[i]['general']['count']
            general_sum_median_h = tracker_d[i]['general']['sum_median_h']
            general_sum_D = tracker_d[i]['general']['sum_D']
            general_sum_bad = tracker_d[i]['general']['sum_bad']

            avg_diameter = general_sum_D / general_count
            bad_quality_percentage = (general_sum_bad / general_count) * 100

            text = f"Яблоко {int(i)} вышло на {line} линии!\n"
            text += f"Средний h - {round(general_sum_median_h / general_count, 2)}\n"
            text += f"Средний диаметр - {round(avg_diameter, 2)}\n"
            text += f"Качество: {round(bad_quality_percentage, 2)}% плохих кадров\n"

            print(text)

    end = time.time()
    video.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\nСреднее время обработки одного к/c - {(end - start) / frame_count}")

if __name__ == "__main__":
    tracker_model = YOLO(tracker_path)
    video = cv2.VideoCapture(video_path)

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    main()
