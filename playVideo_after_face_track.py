import cv2
import numpy as np

def track_face():
    face_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_alt.xml')
    
    # Start default camera
    cap = cv2.VideoCapture(0) 
    scaling_factor = 0.5
    # video_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    num_frame_detected = 0

    while True:
        num_face_detect = 0
        ret, frame = cap.read()
        frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in face_rects:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 3)
            num_face_detect += 1
            # print num_face_detect  # count how many face detected

            if not ret:
                break
            num_frame_detected += 1 # counter for face detect frame

        # cv2.imshow('Face Detector', frame)
        c = cv2.waitKey(1)
        if c == 27:
            break

        
        if num_frame_detected == 40:
            play_video()
            num_frame_detected = 0
            show_img()

    # count number of frames during face detection
    # print num_frame_detected  


    # cap.release()
    # cv2.destroyAllWindows()


def play_video():
    cap = cv2.VideoCapture('test.avi')
 
    if (cap.isOpened() == False): 
      print("Error opening video stream or file")
     
    while(cap.isOpened()):
      ret, frame = cap.read()
      if ret == True:
     
        cv2.imshow('clip',frame)

        if cv2.waitKey(75) & 0xFF == ord('q'):
          break

      else: 
        break

    cap.release()
    cv2.destroyAllWindows()

def show_img():

    img = cv2.imread('panda.jpg')
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Gray Image', img)
    cv2.waitKey(1500)
    cv2.destroyAllWindows()

if __name__=='__main__':

    track_face()

