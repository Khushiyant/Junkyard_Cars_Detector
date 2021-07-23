import cv2 as cv
import numpy as np
from tracker import *


def findObjects(outputs, img, confThreshold, nmsThreshold):   # DETECTION AND TRACKING
    counter = 0
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w, h = int(det[2]*wT), int(det[3]*hT)
                x, y = int((det[0]*wT)-w/2), int((det[1]*hT)-h/2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))

    indices = cv.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)
    detections = []
    for i in indices:
        i = i[0]
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        if (y+h//2) in [199,200]:
            counter += 1
        detections.append([x, y, w, h])
    tracker = EuclideanDistTracker()
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cv.putText(img, str(id), (x, y + 15),
                   cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
        cv.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
        cv.circle(img, (x+w//2, y+h//2), 2, (0, 0, 255), 5)
    return counter


def main(path="Resources\\Drone.mp4"):
    # INITIALISATION

    cap = cv.VideoCapture(path)
    whT = 320
    confThreshold = 0.6
    nmsThreshold = 0.2

    # LOAD MODEL

    classesFile = "Resources\\coco.names"    # Classes names file
    classNames = []
    with open(classesFile, 'rt') as f:
        classNames = f.read().rstrip('n').split('n')

    # MODEL SETUP

    modelConfiguration = "Resources\\custom-yolov4-tiny-detector.cfg"
    modelWeights = "Resources\\custom-yolov4-tiny-detector_best.weights"
    net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

    counter = 0 # Car Counter
    while cap.isOpened():
        try:
            _, tmp = cap.read()
            img = cv.resize(tmp, (0, 0), fx=0.6, fy=0.6)
        except Exception as e:
            img = tmp.copy()

        # PUTTING TEXT AND LINE

        cv.putText(img, "Khushiyant", (520, 60),
                   cv.FONT_HERSHEY_DUPLEX, 1.3, (255, 255, 255), 2)
        cv.putText(img, str(counter), (1030, 60),
                   cv.FONT_HERSHEY_DUPLEX, 1.3, (255, 255, 255), 2)
        cv.line(img, (0, 200), (1200, 200), (0, 0, 0), 2)  # Black

        # SETTING PARAMETERS

        blob = cv.dnn.blobFromImage(
            img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
        net.setInput(blob)
        layersNames = net.getLayerNames()
        outputNames = [(layersNames[i[0] - 1])
                       for i in net.getUnconnectedOutLayers()]
        outputs = net.forward(outputNames)

        # FUNCTION CALLING

        counter += findObjects(outputs, img, confThreshold, nmsThreshold)
        # print(counter)

        # RESULT

        cv.imshow('Final Result', img)
        key = cv.waitKey(5)
        if  key == 27:
            break
