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
        self.EyeArThresh = 0.35  # Recomennded value: 0.32 for video, 0.31 for camera
        self.EyeArConsecFrame = 3  # Recomennded value: 3 for video, 3 for camera
        self.COUNTER = 0
        self.TOTAL = 0
        self.initCameraParameters = True
        self.earValues = []
        self.controlCounter = 0

    def EyeAspectRatioMethod(self, eye):
        # vertical
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        # horizontal
        C = dist.euclidean(eye[0], eye[3])
        # eye aspect ratio
        earCompute = (A + B) / (2.0 * C)
        # return the eye aspect ratio:
        return earCompute

    def CreateCommandLineArguments(self):
        ap = argparse.ArgumentParser()
        ap.add_argument("-p", "--shape-predictor", required=True, help="Path to facial landmark predictor")
        ap.add_argument("-v", "--video", type=str, default="", help="Path to input video file")
        args = vars(ap.parse_args())
        return args

    def DefineFileType(self, fileOption, args):
        # FILE OPTION DEFINES THE TYPE OF FILE SOURCE: 1 - Video | 2 - Camera
        print("[INFO] starting video stream thread...")
        if (fileOption == 1):
            self.vs = FileVideoStream(args["video"]).start()
            self.fileStream = True
        elif (fileOption == 2):
            self.vs = VideoStream(src=0).start()
            self.fileStream = False
        time.sleep(1.0)

    def InitFaceDetector(self, fileOption):
        # FILE OPTION DEFINES THE TYPE OF FILE SOURCE: 1 - Video | 2 - Camera
        print("[INFO] loading facial landmark predictor...")
        args = self.CreateCommandLineArguments()
        self.detector = dlib.get_frontal_face_detector()
        # READ ME: Zaleceane jezeli chcemy korzystac z roznych predyktorów:
        self.predictor = dlib.shape_predictor(args["shape_predictor"])
        (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        self.DefineFileType(fileOption, args)

    def SetCameraParameters(self,frame,shape):
        leftWidth1X, leftWidth1Y = shape[36]
        leftWidth2X, leftWidth2Y = shape[39]

        leftHeight1X, leftHeight1Y = shape[37]
        leftHeight2X, leftHeight2Y = shape[41]

        leftHeight3X, leftHeight3Y = shape[38]
        leftHeight4X, leftHeight4Y = shape[40]

        rightWidth1X, rightWidth1Y = shape[42]
        rightWidth2X, rightWidth2Y = shape[45]

        rightHeight1X, rightHeight1Y = shape[43]
        rightHeight2X, rightHeight2Y = shape[47]

        rightHeight3X, rightHeight3Y = shape[44]
        rightHeight4X, rightHeight4Y = shape[46]

        earL = ((leftHeight1Y - leftHeight2Y) + (leftHeight3Y - leftHeight4Y)) / (2 * (leftWidth1X - leftWidth2X))
        earR = ((rightHeight1Y - rightHeight2Y) + (rightHeight3Y - rightHeight4Y)) / (2 * (rightWidth1X - rightWidth2X))
        self.ear = (earL + earR) / 2

        for i in range(0, 100):
            self.earValues.append(self.ear)
            time.sleep(0.02)
        self.earValues.sort()
        minEarValues = self.earValues[0:20]
        sum = 0
        for value in minEarValues:
            sum += value
        sum /= len(minEarValues)
        print(f'WARTOSC:{sum}')
        self.EyeArThresh = sum + (sum * 0.1)
        self.initCameraParameters = False
        # self.ear = (leftEAR + rightEAR) / 2.0

        # AVG for both eyes
        # self.ear = (leftEAR + rightEAR) / 2.0

        # Visualization
        # leftEyeHull = cv2.convexHull(leftEye)
        # rightEyeHull = cv2.convexHull(rightEye)
        # cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        # cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        # cv2.circle(frame, (leftWidth1X, leftWidth1Y), 1, (0, 255, 0), 2)
        # cv2.circle(frame, (leftWidth2X, leftWidth2Y), 1, (0, 255, 0), 2)
        #
        # cv2.circle(frame, (leftHeight1X, leftHeight1Y), 1, (0, 255, 0), 2)
        # cv2.circle(frame, (leftHeight2X, leftHeight2Y), 1, (0, 255, 0), 2)
        #
        # cv2.circle(frame, (leftHeight3X, leftHeight3Y), 1, (0, 255, 0), 2)
        # cv2.circle(frame, (leftHeight4X, leftHeight4Y), 1, (0, 255, 0), 2)
        #
        # cv2.circle(frame, (rightWidth1X, rightWidth1Y), 1, (0, 255, 0), 2)
        # cv2.circle(frame, (rightWidth2X, rightWidth2Y), 1, (0, 255, 0), 2)
        #
        # cv2.circle(frame, (rightHeight1X, rightHeight1Y), 1, (0, 255, 0), 2)
        # cv2.circle(frame, (rightHeight2X, rightHeight2Y), 1, (0, 255, 0), 2)
        #
        # cv2.circle(frame, (rightHeight3X, rightHeight3Y), 1, (0, 255, 0), 2)
        # cv2.circle(frame, (rightHeight4X, rightHeight4Y), 1, (0, 255, 0), 2)

    def Run(self):
        # ARGUMENT BELOW DEFINES THE TYPE OF FILE SOURCE:
        self.InitFaceDetector(2)  # 1 - Video | 2 - Camera
        self.control = True
        # while self.control:
        #     if self.fileStream and not self.vs.more():
        #         break
        #     frame = self.vs.read()
        #     frame = imutils.resize(frame, width=450)
        #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #     rects = self.detector(gray, 0)
        #     for rect in rects:
        #         shape = self.predictor(gray, rect)
        #         shape = face_utils.shape_to_np(shape)
        #
        #         for i in range(0, 100):
        #             leftWidth1X, leftWidth1Y = shape[36]
        #             leftWidth2X, leftWidth2Y = shape[39]
        #
        #             leftHeight1X, leftHeight1Y = shape[37]
        #             leftHeight2X, leftHeight2Y = shape[41]
        #
        #             leftHeight3X, leftHeight3Y = shape[38]
        #             leftHeight4X, leftHeight4Y = shape[40]
        #
        #             rightWidth1X, rightWidth1Y = shape[42]
        #             rightWidth2X, rightWidth2Y = shape[45]
        #
        #             rightHeight1X, rightHeight1Y = shape[43]
        #             rightHeight2X, rightHeight2Y = shape[47]
        #
        #             rightHeight3X, rightHeight3Y = shape[44]
        #             rightHeight4X, rightHeight4Y = shape[46]
        #
        #             earL = ((leftHeight1Y - leftHeight2Y) + (leftHeight3Y - leftHeight4Y)) / (
        #                     2 * (leftWidth1X - leftWidth2X))
        #             earR = ((rightHeight1Y - rightHeight2Y) + (rightHeight3Y - rightHeight4Y)) / (
        #                     2 * (rightWidth1X - rightWidth2X))
        #             self.ear = (earL + earR) / 2
        #
        #             self.earValues.append(self.ear)
        #             time.sleep(0.01)
        #
        #         self.earValues.sort()
        #         minEarValues = self.earValues[0:20]
        #         sum = 0
        #         for value in minEarValues:
        #             sum += value
        #         sum /= len(minEarValues)
        #         print(f'WARTOSC:{sum}')
        #         self.EyeArThresh = sum + (sum * 0.1)
        #         self.control = False
        #
        # #self.vs.release()
        # cv2.destroyAllWindows()
        # self.vs.stop()
        # print("[INFO] Koniec inicjalizacji EAR")
        while True:
            # Checking the file video stream
            if self.fileStream and not self.vs.more():
                break
            frame = self.vs.read()
            frame = imutils.resize(frame, width=450)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces:
            rects = self.detector(gray, 0)

            for rect in rects:
                shape = self.predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)
                # Extract left and right eye coordinates then compute EAR for both
                # leftEye = shape[self.lStart:self.lEnd]
                # rightEye = shape[self.rStart:self.rEnd]
                # leftEAR = self.EyeAspectRatioMethod(leftEye)
                # rightEAR = self.EyeAspectRatioMethod(rightEye)

                if (self.control):

                    leftWidth1X, leftWidth1Y = shape[36]
                    leftWidth2X, leftWidth2Y = shape[39]

                    leftHeight1X, leftHeight1Y = shape[37]
                    leftHeight2X, leftHeight2Y = shape[41]

                    leftHeight3X, leftHeight3Y = shape[38]
                    leftHeight4X, leftHeight4Y = shape[40]

                    rightWidth1X, rightWidth1Y = shape[42]
                    rightWidth2X, rightWidth2Y = shape[45]

                    rightHeight1X, rightHeight1Y = shape[43]
                    rightHeight2X, rightHeight2Y = shape[47]

                    rightHeight3X, rightHeight3Y = shape[44]
                    rightHeight4X, rightHeight4Y = shape[46]

                    earL = ((leftHeight1Y - leftHeight2Y) + (leftHeight3Y - leftHeight4Y)) / (
                            2 * (leftWidth1X - leftWidth2X))
                    earR = ((rightHeight1Y - rightHeight2Y) + (rightHeight3Y - rightHeight4Y)) / (
                            2 * (rightWidth1X - rightWidth2X))
                    self.ear = (earL + earR) / 2

                    self.earValues.append(self.ear)
                    time.sleep(0.01)
                    self.controlCounter += 1

                    if (self.controlCounter > 200):
                        self.earValues.sort()
                        print(self.earValues)
                        minEarValues = self.earValues[0:(int(len(self.earValues) * 0.25))]
                        sum = 0
                        for value in minEarValues:
                            sum += value
                        sum /= len(minEarValues)
                        print(f'WARTOSC:{sum}')
                        self.EyeArThresh = sum + (sum * 0.3)
                        self.control = False

                leftEye = shape[self.lStart:self.lEnd]
                rightEye = shape[self.rStart:self.rEnd]
                leftEAR = self.EyeAspectRatioMethod(leftEye)
                rightEAR = self.EyeAspectRatioMethod(rightEye)

                # AVG for both eyes
                self.ear = (leftEAR + rightEAR) / 2.0

                if self.ear < self.EyeArThresh:
                    self.COUNTER += 1
                    # print(self.COUNTER)
                    # if (self.COUNTER > 1000):
                    #     print("[WARNING] KIEROWCA MOGL ZASNAC!")
                else:
                    if (self.COUNTER >= self.EyeArConsecFrame):
                        self.TOTAL += 1
                    self.COUNTER = 0

                # Draw number of blinks and computed EAR for the frame
                cv2.putText(frame,"Blinks: {}".format(self.TOTAL), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                cv2.putText(frame,"EAR: {:.2f}".format(self.ear), (300,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                cv2.putText(frame,"Eye Ar Threash: {:.2f}".format(self.EyeArThresh), (10,310), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

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
