# Author: Siddharth Palani Natarajan 

# ME 468 - Traffic Sign Detection for Autonomous Vehicle

#########################################################

import numpy as np
import cv2
import pickle
# import rclpy
# from rclpy.node import Node

""" Vehicle Speed """

vehicle_speed = str(45)

""" Camera Sensor Resolution (Chrono Camera Sensor Resolution """

framewidth = 640
frameheight = 480
brightness = 300

# Staring Camera Sensor
cap = cv2.VideoCapture(0)  # Chrono - Connect Chrono Camera Sensor #
cap.set(3, framewidth)
cap.set(4, frameheight)
cap.set(10, brightness)
assert cap.isOpened()

# Import Trained
picklepath = './Trained_model.p'
pickle_in = open(picklepath, 'rb')
print('Check1')
print(pickle_in)
print('Check2')
model = pickle.load(pickle_in)

# Probability Threshold
threshold = 0.75

# Labelling Font
font = cv2.FONT_HERSHEY_TRIPLEX

""" Image Processing """

def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def equalize(img):
    img = cv2.equalizeHist(img)
    return img

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255
    return img

def getClassName(classNo):
    if classNo == 1:
        return '30'  # Speed Limit 30 km/hr
    elif classNo == 3:
        return '60'  # Speed Limit 30 km/hr
    elif classNo == 14:
        return '0'  # STOP

while True:

    # Read Image
    success, imgOrignal = cap.read()

    # Image Processing
    img = np.asarray(imgOrignal)
    img = cv2.resize(img, (32, 32))
    img = preprocessing(img)
    # cv2.imshow("Processed Image", img)
    img = img.reshape(1, 32, 32, 1)
    cv2.putText(imgOrignal, "Traffic Sign:", (20, 35), font, 0.75, (255, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(imgOrignal, "Probability:", (20, 75), font, 0.75, (255, 0, 0), 1, cv2.LINE_AA)

    # Image Prediction
    predictions = model.predict(img)
    classIndex = model.predict_classes(img)
    probabilityValue = np.amax(predictions)

    if probabilityValue > threshold:
        cv2.putText(imgOrignal, str(getClassName(classIndex)), (190, 35), font, 0.75, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(imgOrignal, str(round(probabilityValue*100, 2)) + "%", (180, 75), font, 0.75, (255, 0, 0), 1, cv2.LINE_AA)

        # ROS2 - To be published to the Perception node for Planning #
        vehicle_speed = str(getClassName(classIndex))
        print(vehicle_speed)
    else:
        print(vehicle_speed)

    cv2.imshow("Result", imgOrignal)

    if cv2.waitKey(1) and 0xFF == ord('q'):
        break






