import cv2

# Подключаемся к камере через RTSP URL
rtsp_url = "rtsp://admin:147258pf@192.168.1.25:554/Streaming/channels/1"
rtsp_url="rtsp://admin:147258pf@192.168.1.64:554/ISAPI/Streaming/Channels/101"
cap = cv2.VideoCapture(rtsp_url)  # Принудительно используем ffmpeg

if not cap.isOpened():
    print("Не удалось подключиться к камере")
else:
    print("Подключение установлено")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Не удалось получить кадр")
        break

    cv2.imshow("IP Camera Stream", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()