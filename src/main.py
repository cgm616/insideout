import cv2
import numpy as np
from threading import Thread



# This class and most of the code that interacts with it was found at
# https://www.pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/
class WebcamVideoStream:
    def __init__(self, src=0):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
 
        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self
 
    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return
 
            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()
 
    def read(self):
        # return the frame most recently read
        return self.frame
 
    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True


class MoodRing:
    def __init__(self):
        self.cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.webcam = WebcamVideoStream(src=0).start()

    def run(self):
        # Loop until stopped
        while True:
            # Get the current frame from the webcam
            frame = self.webcam.read()
            
            scale_percent = 50 # percent of original size
            width = int(frame.shape[1] * scale_percent / 100)
            height = int(frame.shape[0] * scale_percent / 100) 

            # Convert to greyscale and downscale
            grey = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (width, height))

            # Using the cascade, identify all faces in the image
            faces = self.cascade.detectMultiScale(
                grey,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
                #flags = cv2.CV_HAAR_SCALE_IMAGE
            )

            # Put a rectangle around each face
            for (x, y, w, h) in faces:
                cv2.rectangle(grey, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Show the image
            cv2.imshow("Capturing", grey)

            key = cv2.waitKey(1)

            if key == ord('q'):
                self.webcam.stop()
                cv2.destroyAllWindows()
                break


if __name__ == '__main__':
    app = MoodRing()
    app.run()