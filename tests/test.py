number_video = 1

import time

import cv2


def main():

    # reading the input
    cap = cv2.VideoCapture('rtsp://admin:147258pf@192.168.1.25/Streaming/channels/1/')
    time.sleep(2)
    height, width = 720, 1280
    print(width, height)
    output = cv2.VideoWriter(
        f"output_{number_video}.avi", cv2.VideoWriter_fourcc('M','J','P','G'), 25, (width, height))

    while(True):
        ret, frame = cap.read()
        if(ret):

            # adding rectangle on each frame
            #cv2.rectangle(frame, (100, 100), (500, 500), (0, 255, 0), 3)

            # writing the new frame in output
            output.write(frame)
            cv2.imshow("output", frame)
            if cv2.waitKey(1) & 0xFF == ord(' '):
                break


    output.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()