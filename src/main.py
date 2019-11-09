import cv2
import numpy as np
from threading import Thread
import pyfirmata
import time as time_
import keras

EMOTIONS = ["angry","disgust","scared", "happy", "sad", "surprised","neutral"]

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


class Arduino:
    def __init__(self, serial):
        self.board = pyfirmata.Arduino(serial)
        self.state = [False] * 7
        self.on_intervals = [1] * 7
        millis = int(round(time_.time() * 1000))
        self.times = [millis + 1] * 7
        self.stop = False

    def loop(self):
        while not self.stop:
            millis = int(round(time_.time() * 1000))
            for i in range(len(self.times)):
                if millis >= self.times[i]:
                    if self.state[i]:
                        self.board.digital[i+2].write(0)
                        self.times[i] += 20 - self.on_intervals[i]
                    else:
                        self.board.digital[i+2].write(1)
                        self.times[i] += self.on_intervals[i]
                    self.state[i] = not self.state[i]

    
    def run(self):
        Thread(target=self.loop, args=()).start()
        return self
    
    def update_colors(self, prob):
        for i in range(7):
            self.on_intervals[i] = 1 + int(prob[i] * 18.0)


class MoodRing:
    def __init__(self, file='model/cnn.h5', arduino=False, serial=''):
        self.arduino = arduino
        if arduino:
            print("starting arduino")
            self.board = Arduino(serial).run()
        print("loading model")
        self.model = keras.models.load_model(file)
        print("loading cascade")
        self.cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        print("starting webcam")
        self.webcam = WebcamVideoStream(src=0).start()

    def crop_found_faces(self, image, locations):
        return np.array([
            cv2.normalize(
                cv2.resize(image[y:y+h, x:x+w], (48, 48)),
                None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            for (x, y, w, h) in locations])

    def render_to_screen(self, image, locations, predictions):
        # Put a rectangle around each face
        for ((x, y, w, h), vector) in zip(locations, predictions):
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            emotion_probability = np.max(vector)
            label = EMOTIONS[vector.argmax()]
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            
            # Show the image
        cv2.imshow("Capturing", image)

    def render_to_arduino(self, probabilities):
        if len(probabilities) > 0 and self.arduino:
            self.board.update_colors(probabilities[0])

    def run(self):
        # Loop until stopped
        while True:
            # Get the current frame from the webcam
            frame = self.webcam.read()
            
            scale_percent = 50 # percent of original size
            width = int(frame.shape[1] * scale_percent / 100)
            height = int(frame.shape[0] * scale_percent / 100) 

            resized = cv2.resize(frame, (width, height))

            # Convert to greyscale and downscale
            grey = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

            # Using the cascade, identify all faces in the image
            face_locations = self.cascade.detectMultiScale(
                grey,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
                #flags = cv2.CV_HAAR_SCALE_IMAGE
            )

            face_images = self.crop_found_faces(grey, face_locations).reshape(len(face_locations), 48, 48, 1)

            predictions = self.model.predict(face_images)

            self.render_to_screen(resized, face_locations, predictions)
            self.render_to_arduino(predictions)

            key = cv2.waitKey(1)

            if key == ord('q'):
                self.webcam.stop()
                cv2.destroyAllWindows()
                break

    def test_arduino(self):
        print("running")
        self.arduino.run()


if __name__ == '__main__':
    app = MoodRing('model/cnn.h5')
    app.run()