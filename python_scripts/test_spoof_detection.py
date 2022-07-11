import numpy as np
import cv2
from sklearn.externals import joblib


def detect_face(img, faceCascade):
    faces = faceCascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(110, 110)
        #flags = cv2.CV_HAAR_SCALE_IMAGE
    )
    return faces


def calc_hist(img):
    histogram = [0] * 3
    for j in range(3):
        histr = cv2.calcHist([img], [j], None, [256], [0, 256])
        histr *= 255.0 / histr.max()
        histogram[j] = histr
    return np.array(histogram)


if __name__ == "__main__":

    # # Load model
    clf = None
    try:
        clf = joblib.load('trained_models/print-attack_trained_models/print-attack_ycrcb_luv_extraTreesClassifier.pkl')
    except IOError as e:
        print("Error loading model")
        exit(0)

    # Initialize face detector
    cascPath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascPath)

    img_bgr = cv2.imread('test_img/anh_the.jpg')
    # img_bgr = cv2.imread('test_img/real1.jpg')
    # img_bgr = cv2.imread('test_img/20194452.jpg')
    scale_percent = 40 # percent of original size
    width = int(img_bgr.shape[1] * scale_percent / 100)
    height = int(img_bgr.shape[0] * scale_percent / 100)
    dim = (width, height)
  
    # resize image
    img_bgr = cv2.resize(img_bgr, dim, interpolation = cv2.INTER_AREA)
    # img_bgr = cv2.resize(img_bgr, (300, 400))
    
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = detect_face(img_gray, faceCascade)
    
    if len(faces) == 0:
        print('No faces')
    for i, (x, y, w, h) in enumerate(faces):
        roi = img_bgr[y:y+h, x:x+w]

        img_ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCR_CB)
        img_luv = cv2.cvtColor(roi, cv2.COLOR_BGR2LUV)

        ycrcb_hist = calc_hist(img_ycrcb)
        luv_hist = calc_hist(img_luv)

        feature_vector = np.append(ycrcb_hist.ravel(), luv_hist.ravel())
        feature_vector = feature_vector.reshape(1, len(feature_vector))

        prediction = clf.predict_proba(feature_vector)
        prob = prediction[0][1]

        cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (0, 0, 255), 2)
        if prob >= 0.5:
            cv2.putText(img_bgr, 'Spoof', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
        else:
            cv2.putText(img_bgr, 'Live', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
        print(prob)
    
    cv2.imshow('img', img_bgr)
    cv2.waitKey(0)
    
