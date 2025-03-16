import cv2

# Инициализация камеры
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("Ошибка: не удалось открыть камеру.")
    exit()

# Получение параметров видео
frame_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_rate = int(camera.get(cv2.CAP_PROP_FPS))

# Кодек и создание объекта VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('out.mp4', fourcc, frame_rate, (frame_width, frame_height))

print("Нажмите пробел, чтобы остановить запись.")

while True:
    ret, frame = camera.read()
    if not ret:
        print("Ошибка: не удалось получить кадр с камеры.")
        break

    # Отображение видео
    cv2.imshow('Video', frame)

    # Сохранение текущего кадра в файл
    out.write(frame)

    # Остановка при нажатии пробела
    if cv2.waitKey(1) & 0xFF == ord(' '):
        print("Остановка записи.")
        break

# Освобождение ресурсов
camera.release()
out.release()
cv2.destroyAllWindows()