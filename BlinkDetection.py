from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2

class BlinkDetection():
    def __init__(self):
        self.EyeArThresh = 0.28 # Recomennded value: 0.3
        self.EyeArConsecFrame = 3
        self.COUNTER = 0
        self.TOTAL = 0
    def EyeAspectRatioMethod(self,eye):
        # vertical
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])

        # horizontal
        C = dist.euclidean(eye[0],eye[3])

        # eye aspect ratio
        earCompute = (A + B) / (2.0 * C)

        # return the eye aspect ratio:
        return earCompute
    def CreateCommandLineArguments(self):
        ap = argparse.ArgumentParser()
        ap.add_argument("-p","--shape-predictor",required=True,help="Path to facial landmark predictor")
        ap.add_argument("-v", "--video", type=str, default="",help="Path to input video file")
        args = vars(ap.parse_args())
        return args
    def DefineFileType(self,fileOption,args):
        # FILE OPTION DEFINES THE TYPE OF FILE SOURCE: 1 - Video | 2 - Camera
        print("[INFO] starting video stream thread...")
        if (fileOption == 1):
            self.vs = FileVideoStream(args["video"]).start()
            self.fileStream = True
        elif (fileOption == 2):
            self.vs = VideoStream(src=0).start()
            self.fileStream = False
        time.sleep(1.0)
    def InitFaceDetector(self,fileOption):
        # FILE OPTION DEFINES THE TYPE OF FILE SOURCE: 1 - Video | 2 - Camera
        print("[INFO] loading facial landmark predictor...")
        args = self.CreateCommandLineArguments()
        self.detector = dlib.get_frontal_face_detector()
        #self.predictor = dlib.shape_predictor(args["shape_predictor"])
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        self.DefineFileType(2,args)

    def Run(self):
        self.InitFaceDetector(2)
        while True:
            # Checking the file video stream
            if self.fileStream and not self.vs.more():
                break
            frame = self.vs.read()
            frame = imutils.resize(frame, width=450)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces:
            rects = self.detector(gray,0)

            for rect in rects:
                shape = self.predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)
                # Extract left and right eye coordinates then compute EAR for both
                leftEye = shape[self.lStart:self.lEnd]
                rightEye = shape[self.rStart:self.rEnd]
                leftEAR = self.EyeAspectRatioMethod(leftEye)
                rightEAR = self.EyeAspectRatioMethod(rightEye)

                # AVG for both eyes
                ear = (leftEAR + rightEAR) / 2.0

                # Visualization
                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                cv2.drawContours(frame, [leftEyeHull], -1, (0,255,0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0,255,0), 1)

                # Check if the EAR is below the blink threshold
                if ear < self.EyeArThresh:
                    self.COUNTER += 1
                else:
                    if (self.COUNTER >= self.EyeArConsecFrame):
                        self.TOTAL += 1
                    self.COUNTER = 0

                # Draw number of blinks and computed EAR for the frame
                cv2.putText(frame,"Blinks: {}".format(self.TOTAL), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                cv2.putText(frame,"EAR: {:.2f}".format(ear), (300,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            # Show the frame:
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1)
            if key == ord("q"):
                break
        # CLEANUP
        self.vs.releas()
        cv2.destroyAllWindows()
        self.vs.stop()


bp = BlinkDetection()
bp.Run()