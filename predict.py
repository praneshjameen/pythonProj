from tensorflow import keras
from keras.models import load_model
import cv2
import numpy as np
from keras import backend as K
import pyrebase


num_channel=3
model=load_model('Trained_model.hdf5')

config = {
    "apiKey": "AIzaSyDuUe2c0lcsLpahf2zUkxGoXVp6o1-jBFo",
    "authDomain": "wounddetector.firebaseapp.com",
    "databaseURL": "https://wounddetector.firebaseio.com",
    "projectId": "wounddetector",
    "storageBucket": "wounddetector.appspot.com",
    "messagingSenderId": "797552612130"
  }
firebase = pyrebase.initialize_app(config)
storage=firebase.storage()
storage.child("image/wound.jpg").download("wound.jpg")

# Testing a new image
test_image = cv2.imread('wound.jpg')
test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
test_image = cv2.resize(test_image, (128, 128))
test_image = np.array(test_image)
test_image = test_image.astype('float32')
test_image /= 255
print(test_image.shape)

if num_channel == 1:
    if K.image_dim_ordering() == 'th':
        test_image = np.expand_dims(test_image, axis=0)
        test_image = np.expand_dims(test_image, axis=0)
        print(test_image.shape)
    else:
        test_image = np.expand_dims(test_image, axis=3)
        test_image = np.expand_dims(test_image, axis=0)
        print(test_image.shape)
else:
    if K.image_dim_ordering() == 'th':
        test_image = np.rollaxis(test_image, 2, 0)
        test_image = np.expand_dims(test_image, axis=0)
        print(test_image.shape)
    else:
        test_image = np.expand_dims(test_image, axis=0)
        print(test_image.shape)


# Predicting the test image
database=firebase.database()
print((model.predict(test_image)))
classs=model.predict_classes(test_image)
if(classs[0]==0):
    print("phase 1")
    database.child("data").child("woundName").set("Hemostasis")
    database.child("data").child("description").set("Phase 1 - Hemostasis is the process of the wound being closed by clotting. Hemostasis starts when blood leaks out of the body.")
    database.child("data").child("status").set(1)
elif(classs[0]==1):
    print("Phase 2")
    database.child("data").child("woundName").set("Inflammatory")
    database.child("data").child("description").set("Phase 2 - Inflammation is the second stage of wound healing and begins right after the injury when the injured blood vessels leak transudate (made of water, salt, and protein) causing localized swelling.")
    database.child("data").child("status").set(1)
elif(classs[0]==2):
    print("Phase 3")
    database.child("data").child("woundName").set("Proliferative")
    database.child("data").child("description").set("Phase 3 -  In the proliferative phase, the wound contracts as new tissues are built.")
    database.child("data").child("status").set(1)
elif(classs[0]==3):
    print("Phase 4")
    database.child("data").child("woundName").set("Remodelling")
    database.child("data").child("description").set("Final Phase -  The cells that had been used to repair the wound but which are no longer needed are removed by apoptosis, or programmed cell death.")
    database.child("data").child("status").set(1)
print(classs)