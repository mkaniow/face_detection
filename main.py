'''
This is main file that activates webcam and face detect model

If you have more than one webcam you can choose which one you want to activate. To do that you need to change 
value in line 14
'''

import tensorflow as tf
import cv2
import numpy as np

facetracker = tf.keras.models.load_model('facetracker.h5')

cap = cv2.VideoCapture(0) # <-- choose webcam, 0 is default one
while cap.isOpened():
    res , frame = cap.read()

    frame = cv2.resize(frame, (600,600))

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = tf.image.resize(rgb, (128,128))
    
    yhat = facetracker.predict(np.expand_dims(resized/255,0))
    sample_coords = yhat[1][0]
    
    if yhat[0] > 0.6: 
        # Controls the main rectangle
        cv2.rectangle(frame, 
                      tuple(np.multiply(sample_coords[:2], [600,600]).astype(int)),
                      tuple(np.multiply(sample_coords[2:], [600,600]).astype(int)), 
                            (0,255,0), 2)
        
        #blur face
        coord1 = list(np.multiply(sample_coords[:2], [600,600]).astype(int))
        coord2 = list(np.multiply(sample_coords[2:], [600,600]).astype(int))
        blur_x1 = coord1[0]
        blur_y1 = coord1[1]
        blur_x2 = coord2[0]
        blur_y2 = coord2[1]

        roi = frame[blur_y1:blur_y2, blur_x1:blur_x2]
        blur_image = cv2.GaussianBlur(roi,(51,51),0)

        frame[blur_y1:blur_y2, blur_x1:blur_x2] = blur_image
        
        # Controls the text rendered
        cv2.putText(frame, 'debil', tuple(np.add(np.multiply(sample_coords[:2], [600,600]).astype(int),[0,25])),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
    
    cv2.imshow('FaceTrack', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()