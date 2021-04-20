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
# READ ME:
# Uwazam, ze po pierwszym uruchomieniu w zaleznosci od posiadanej kamerki nalezy sprobować skonfigurować
# program pod swoj sprzet. W tym celu polecam sprobowac zmodyfikowac wartosci zmiennych
# EyeArThresh oraz EyeArConsecFrame. Obie zmienne znajduja sie w konstruktorze klasy.
# INSTRUKCJA URUCHOMIENIA
# Jezeli chcemy uzyc przykladowego wideo:
# 1) Zmieniamy w self.InitFaceDetector (linijka 61 ) wartosc arguementu na 1
# 2) W wierszu poleceń odpalamy program: python BlinkDetection.py --shape-predictor shape_predictor_68_face_landmarks.dat --video freeddie.mp4
# Jezeli chcemy uzyc kamery:
# 1) Zmieniamy w self.InitFaceDetector (linijka 61 ) wartosc arguementu na 2
# 2) W wierszu poleceń odpalamy program: python BlinkDetection.py --shape-predictor shape_predictor_68_face_landmarks.dat
class BlinkDetection():
    def __init__(self):
        self.EyeArThresh = 0.30 # Recomennded value: 0.32 for video, 0.31 for camera
        self.EyeArConsecFrame = 3 # Recomennded value: 3 for video, 3 for camera
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
        # READ ME: Zaleceane jezeli chcemy korzystac z roznych predyktorów:
        self.predictor = dlib.shape_predictor(args["shape_predictor"])
        (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        self.DefineFileType(fileOption,args)

    def Run(self):
        # ARGUMENT BELOW DEFINES THE TYPE OF FILE SOURCE:
        self.InitFaceDetector(1) # 1 - Video | 2 - Camera
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
            # Close program
            if key == ord("q") or key == 27:
                break
        # CLEANUP
        self.vs.release()
        cv2.destroyAllWindows()
        self.vs.stop()

# Uruchomienie progrmu:
bp = BlinkDetection()
bp.Run()