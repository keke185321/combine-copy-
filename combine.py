from lib.device import Camera
from lib.processors_noopenmdao import findFaceGetPulse
from lib.interface import plotXY, imshow, waitKey, destroyWindow
from cv2 import moveWindow
import argparse
import numpy as np
import datetime
#TODO: work on serial port comms, if anyone asks for it
#from serial import Serial
import socket
import sys
import cv2
#from emotion_recognition import EmotionRecognition
#from constants import *

from scipy.spatial import distance as dist
#from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
#import playsound
import imutils
import time
import dlib
import Tkinter as tki
from PIL import Image
from PIL import ImageTk

class getPulseApp(object):
    def __init__(self, args):
	self.COUNTER = 0
        self.cameras = []
        self.selected_cam = 0
        for i in range(3):
            camera = Camera(camera=i)  # first camera by default
            if camera.valid or not len(self.cameras):
                self.cameras.append(camera)
            else:
                break
        self.processor = findFaceGetPulse(bpm_limits=[50, 160],
                                          data_spike_limit=2500.,
                                          face_detector_smoothness=10.)

        self.plot_title = "Real-Time Heart Rate"
    def eye_aspect_ratio(self,eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])

	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)

	# return the eye aspect ratio
	return ear


    def main_loop(self,eye_aspect_ratio,root):
        """
        Single iteration of the application's main loop.
        """
        # Get current image frame from the camera
        frame = self.cameras[self.selected_cam].get_frame()
	#print frame.shape
        self.h, self.w, _c = frame.shape
	#print self.h, self.w, _c
	#result = network.predict(poc.format_image(frame))
	#frame = cameras[selected_cam].get_frame()
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# detect faces in the grayscale frame
	rects = detector(gray, 0)
	ALARM_ON=False
	# loop over the face detections
	for rect in rects:
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)
		ear = (leftEAR + rightEAR) / 2.0
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		#cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		#cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
		if ear < EYE_AR_THRESH:
			self.COUNTER += 1
			if self.COUNTER >= EYE_AR_CONSEC_FRAMES:
				# if the alarm is not on, turn it on
				if not ALARM_ON:
					print ear, 'sleepy'
					ALARM_ON = True
		else:
			print ear, 'awake'
			self.COUNTER = 0
			ALARM_ON = False
	self.processor.frame_in = frame
	# process the image frame to perform all needed analysis
	self.processor.run(self.selected_cam,ALARM_ON,root)
	# collect the output frame for display
	output_frame = self.processor.frame_out
	img = Image.fromarray(output_frame)
	imgtk = ImageTk.PhotoImage(image=img)
	lmain.imgtk = imgtk 
	lmain.configure(image=imgtk)
	return output_frame

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Webcam pulse detector.')
    parser.add_argument('--serial', default=None,
                        help='serial port destination for bpm data')
    parser.add_argument('--baud', default=None,
                        help='Baud rate for serial transmission')
    parser.add_argument('--udp', default=None,
                        help='udp address:port destination for bpm data')
    parser.add_argument('--train', default=None,
                        help='udp address:port destination for bpm data')
    args = parser.parse_args()
    App = getPulseApp(args)
    font = cv2.FONT_HERSHEY_SIMPLEX
    EYE_AR_THRESH = 0.3
    EYE_AR_CONSEC_FRAMES = 48
    
    ALARM_ON = False
    print("[INFO] loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    # grab the indexes of the facial landmarks for the left and
    # right eye, respectively
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"] 
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    te=[]
    root = tki.Tk()
    root.bind('<Escape>', lambda e: root.quit())
    root.wm_title("Pulse Sleep Detection")
    lmain = tki.Label(root)
    lmain.pack()
    lmain1 = tki.Label(root)
    lmain1.pack( side = tki.LEFT )
    while True:
        App.main_loop(App.eye_aspect_ratio,root)

    root.mainloop()
    
    
