import logging
import cv2
import time
from config import *
import tensorflow as tf
from tensorflow.keras.applications.resnet import preprocess_input
import numpy as np
from ultralytics import YOLO
import os
import torch

# Проверка доступности GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

logging.getLogger('ultralytics').setLevel(logging.WARNING)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("TensorFlow использует GPU")
else:
    print("TensorFlow не использует GPU")

time.sleep(2)

def prepare_image(img_array, target_size=(760, 760)):
    img_array = tf.image.resize(img_array, target_size)  # Изменение размера
    img_array = np.expand_dims(img_array, axis=0)  # Добавляем размерность для батча
    img_array = np.copy(img_array)
    img_array = preprocess_input(img_array)  # Применяем ту же предварительную обработку
    return img_array


def draw_custom_annotations(frame, results, tracker_d, save_dir, frame_count):
    output = []
    for obj in results[0].boxes:
        x1, y1, x2, y2 = map(int, obj.xyxy[0].tolist())

        D = max(abs(x2 - x1), abs(y2 - y1))
        track_obj = obj.id
        if not track_obj is None:
            track_id = track_obj.item()
            # print(track_obj)
            cropped_object = frame[y1:y2, x1:x2]

            # Классифицируем

            start_clas = time.time()
            # class_results = classification_model.predict(prepare_image(cropped_object))
            #
            # class_label = ['bad', 'good'][np.argmax(class_results)]
            # probs = max(class_results[0])
            # TF class_results = classification_model.predict(cropped_object, conf=0.25)
            # class_label = class_results[0].names[np.argmax(class_results[0].probs.data).item()]
            # TF class_label = class_results[0].names[np.argmax(class_results[0].probs.cpu().data).item()]

            # probs = max(class_results[0].probs.data).item()
            probs = 0.9
            class_label = "good"

            # Установите нужный текст для аннотации
            custom_text = f'{int(track_id)}' # {class_label} {round(probs, 2)} D={round(D, 2)}'

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
                # os.makedirs(path_to_save, exist_ok=True)

            # object_filename = os.path.join(path_to_save, f'id_{track_id}_label_{class_label}_frame_{frame_count}.jpg')
            # cv2.imwrite(object_filename, cropped_object)


            hsv_frame = cv2.cvtColor(cropped_object, cv2.COLOR_BGR2HSV)
            h_channel, s_channel, v_channel = cv2.split(hsv_frame)

            # Вычисление медианы для каждого канала
            median_h = np.median(h_channel)
            median_s = np.median(s_channel)
            median_v = np.median(v_channel)

            tracker_d[track_id]['history'].append({
                'label': class_label,
                'prob': probs,
                'D': round(D, 2),
                'x1': x1,
                'x2': x2,
                'y1': y1,
                'y2': y2,
                'median_h': median_h,
                'median_s': median_s,
                'median_v': median_v
            })
            tracker_d[track_id]['general']['count'] += 1
            tracker_d[track_id]['general']['sum_bad'] += int(class_label != 'good')
            tracker_d[track_id]['general']['sum_D'] += round(D, 2)
            tracker_d[track_id]['general']['sum_median_h'] += median_h

            if y2 < 30:
                if x2 > 700:
                    tracker_d[track_id]['general']['line'] = 1
                else:
                    tracker_d[track_id]['general']['line'] = 2
                output.append(track_id)

            # Определите цвет и шрифт
            if class_label == "good":
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 2
            text_size, _ = cv2.getTextSize(custom_text, font, font_scale, font_thickness)
            text_x = x1
            text_y = y1 - 10 if y1 - 10 > 10 else y1 + text_size[1] + 10

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, custom_text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)
    return tracker_d, output


def main():
    # Настройка выходного видео
    # ROI
    # r = (422, 1, 119, 393)  # (225, 37, 87, 365)

    # frame_width = r[2]
    # frame_height = r[3]
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_file = f'Yolo_tracker_gpu.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, 8.0, (frame_width, frame_height))

    # Директория для сохранения вырезанных объектов
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

        # frame = frame[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
        # frame = cv2.resize(frame, None, fx=0.2, fy=0.2)
        # Применяем модель YOLO для детекции объектов
        results = tracker_model.track(frame, persist=True, conf=0.3)
        tracker_d, output = draw_custom_annotations(frame, results, tracker_d, img_save_path, frame_count)

        # Записываем кадр в выходное видео
        # out.write(frame)
        cv2.imshow('out', frame)
        out.write(frame)
        key = cv2.waitKey(1)
        if key == ord(" "):
            r = cv2.selectROI("Select ROI", frame)
            print(r)

            # crop the image to the selected region (ROI)
            roi_image = frame[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
        for i in output:
            line = tracker_d[i]['general']['line']
            general_sum_bad = tracker_d[i]['general']['sum_bad']
            general_count = tracker_d[i]['general']['count']
            general_sum_D = tracker_d[i]['general']['sum_D']
            general_sum_median_h = tracker_d[i]['general']['sum_median_h']

            text = f"Яблоко {int(i)} вышло на {line} линии!\n" + ' ' * 4
            # text += f"Плохое в {general_sum_bad}/{general_count} ({round(100 * general_sum_bad / general_count)}%) кадрах\n" + ' ' * 4
            text += f"Средний диаметр - {round(general_sum_D / general_count, 2)}\n" + ' ' * 4
            text += f"Средний h - {round(general_sum_median_h / general_count, 2)}\n" + ' ' * 4

            print(text)

        # Отображаем кадр (для отладки, если нужно)
        # cv2.imshow(window_name, frame)

    end = time.time()
    video.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\nСреднее время обработки одного к/c - {(end - start) / frame_count}")


if __name__ == "__main__":
    tracker_model = YOLO(tracker_path)
    classification_model = YOLO(classification_model_path)
    video = cv2.VideoCapture(video_path)

    main()
