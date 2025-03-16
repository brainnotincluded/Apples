import time

import cv2

# Укажите URL вашего RTSP-потока
ip_address = "192.168.1.24"  # IP адрес камеры
username = "admin"
password = "147258pf"

# Создание URL для подключения к RTSP-потоку
# Проверьте, поддерживает ли ваша камера RTSP-поток; RTSP-ссылка может отличаться.
rtsp_url = f"rtsp://admin:147258pf@192.168.1.24/Streaming/channels/1"
print(rtsp_url)


# Создаем объект VideoCapture с RTSP-ссылкой и настраиваем параметры
cap = cv2.VideoCapture(rtsp_url)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Устанавливаем буфер для минимальной задержки
time.sleep(5)
# Проверяем, что поток доступен
if not cap.isOpened():
    print("Ошибка: не удалось открыть поток.")
    exit()

# Основной цикл захвата и отображения
while True:
    # Пропускаем все кадры в буфере, чтобы получить последний
    while cap.grab():
        pass

    # Чтение и отображение последнего кадра
    ret, frame = cap.retrieve()
    if not ret:
        print("Ошибка: не удалось получить кадр.")
        break

    # Отображаем кадр
    cv2.imshow("RTSP Stream", frame)

    # Выход по нажатию клавиши "q"
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Освобождаем ресурсы
cap.release()
cv2.destroyAllWindows()
