import os
import cv2
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import h5py as h5
from PIL import Image
from keras.preprocessing import image
from keras.utils.image_utils import load_img, img_to_array 
from keras.models import load_model

# load model
model = load_model("best_model.h5")

face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

print("w - webcam / i - image")
mode = input("What's your choice? ")

# Captura de Webcam
cap = cv2.VideoCapture(0)

if mode == "W" or mode == "w":
    while True:
        ret, test_img = cap.read()  # captura os frames e retorna o valor booleano e a imagem capturada
        if not ret:
            continue
        gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
    
        faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)
    
        for (x, y, w, h) in faces_detected:
            cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
            roi_gray = gray_img[y:y + w, x:x + h]  # Regiao de corte de interesse, ou seja, area da face da imagem
            roi_gray = cv2.resize(roi_gray, (224, 224))
            img_pixels = img_to_array(roi_gray)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            img_pixels /= 255
    
            pred = model.predict(img_pixels)
    
            # find max indexed array 
            max_index = np.argmax(pred[0])
    
            emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
            predicted_emotion = emotions[max_index]
    
            cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
        resized_img = cv2.resize(test_img, (1000, 700))
        cv2.imshow('Emotion Detection (Press q to quit)', resized_img)
    
        if cv2.waitKey(10) == ord('q'):  # Espera ate a tecla Q ser pressionada
            break
elif mode == "i" or mode == "I":    
   #Input image
    path = "train/Angry/download.jpg"
    img = load_img(path, target_size=(224,224) )
    i = img_to_array(img)/255
    input_arr = np.array([i])
    input_arr.shape
    
    pred = np.argmax(model.predict(input_arr))
    emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
    predicted_emotion = emotions[pred]
    
    print(f" A emocao da imagem e de {predicted_emotion}")
    
    #Interface do OpenCV
    cv2.imshow('Emotion Detection',input_arr[0])
    #Interface do Matplotlib
    plt.imshow(input_arr[0])
    plt.title("Emotion Detection")
    plt.show()
   

cap.release()
cv2.destroyAllWindows
