from random import choice
from numpy import load
from numpy import expand_dims
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from matplotlib import pyplot
from numpy import load
from numpy import expand_dims
from numpy import asarray
from numpy import savez_compressed
from keras.models import load_model
from os import listdir
from os.path import isdir
from PIL import Image
from matplotlib import pyplot
from numpy import savez_compressed
from numpy import asarray
from mtcnn.mtcnn import MTCNN
import pickle
import numpy as np
import cv2
from time import sleep
from datetime import datetime

# def extract_face(image, detector, required_size=(160, 160)):
#     pixels = asarray(image)
#     # detect faces in the image
#     results = detector.detect_faces(pixels)
#     face_array = np.array([])
#     x1 = y1 = x2 = y2 = 0
#     if (len(results) > 0):
#         # extract the bounding box from the first face
#         x1, y1, width, height = results[0]['box']
#         # bug fix
#         x1, y1 = abs(x1), abs(y1)
#         x2, y2 = x1 + width, y1 + height
#         # extract the face
#         face = pixels[y1:y2, x1:x2]
#         # resize pixels to the model size
#         image = Image.fromarray(face)
#         image = image.resize(required_size)
#         face_array = asarray(image)
#     return face_array, x1, y1, x2, y2

# get the face embedding for one face
def get_embedding(model, face_pixels):
    # scale pixel values
    face_pixels = face_pixels.astype('float32')
    # standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # transform face into one sample
    samples = expand_dims(face_pixels, axis=0)
    # make prediction to get embedding
    yhat = model.predict(samples)
    return yhat[0]

logfile_path = './logs/' + str(datetime.now()) + '.log'
logfile = open(logfile_path, 'w')
def log(msg):
    print(f'\n [INFO] {msg}', file=logfile)
    print('\n [INFO] ' + msg)
    return 0

# load the facenet model
emb_model = load_model('facenet_keras.h5')
log('Embedding model loaded')

model = pickle.load(open("svc_model.sav", 'rb'))
log('Classification model loaded')

# create the detector, using default weights
face_detector = MTCNN()
log('Face detector created')

out_encoder = LabelEncoder()
out_encoder.classes_ = np.load('classes.npy')

video_capture = cv2.VideoCapture('video_test.avi')
# get input video size
video_width = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
video_height = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

font = cv2.FONT_HERSHEY_DUPLEX
proba_threshold = 90
required_size=(160, 160)
ret = True
frame_number = 0
while ret:
    if not video_capture.isOpened():
        log('Unable to load camera.')
        sleep(5)
        pass

    ret, frame = video_capture.read()
    if ret is False:
        log('Video ended')
        continue
    frame_number += 1
    # gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    pixels = asarray(frame)
    results = face_detector.detect_faces(pixels)
    if (len(results) > 0):
        for face in results:
            left, top, width, height = face['box']
            # bug fix
            left, top = abs(left), abs(top)
            right, bottom = left + width, top + height
            # extract the face
            face = pixels[top:bottom, left:right]
            # resize pixels to the model size
            image = Image.fromarray(face)
            image = image.resize(required_size)
            face_image = asarray(image)
            # make prediction for face in each frame
            face_emb = get_embedding(emb_model, face_image)
            samples = expand_dims(face_emb, axis=0)
            yhat_class = model.predict(samples)
            yhat_prob = model.predict_proba(samples)
            class_index = yhat_class[0]
            class_probability = yhat_prob[0,class_index] * 100
            predict_name = 'Unknown'
            if class_probability > proba_threshold:
                predict_name = out_encoder.inverse_transform(yhat_class)
            log('In frame ' + str(frame_number) + ', recognized ' + predict_name[0] + ' with probability ' + str(class_probability))
            # draw prediction result onto frame
            cv2.rectangle(frame, (left,top), (right,bottom), (0,255,0), 2)
            cv2.putText(frame, predict_name[0], (left, top - 6), font, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, str(int(class_probability)) + "%", (left, bottom + 12), font, 0.5, (255, 255, 255), 1)
        # save every frames with detected face to image ---> Do not use in production
        cv2.imwrite('./ati_frames/frame' + str(frame_number) + '.jpg', frame)
    else:
        log('No image in frame ' + str(frame_number))
        # save every frames without detected face to image ---> Do not use in production
        cv2.imwrite('./ati_frames/no_face/frame' + str(frame_number) + '.jpg', frame)

    # face_image, left, top, right, bottom = extract_face(frame, face_detector)
    # if (face_image.size > 0):
    #     # make prediction for face in each frame
    #     face_emb = get_embedding(emb_model, face_image)
    #     samples = expand_dims(face_emb, axis=0)
    #     yhat_class = model.predict(samples)
    #     yhat_prob = model.predict_proba(samples)
    #     class_index = yhat_class[0]
    #     class_probability = yhat_prob[0,class_index] * 100
    #     predict_names = out_encoder.inverse_transform(yhat_class)
    #     log('In frame ' + str(frame_number) + ', recognized ' + predict_names[0] + ' with probability ' + str(class_probability))

    #     # draw prediction result onto frame
    #     cv2.rectangle(frame, (left,top), (right,bottom), (0,255,0), 2)
    #     cv2.putText(frame, predict_names[0], (left, top - 6), font, 0.5, (255, 255, 255), 1)
    #     cv2.putText(frame, str(int(class_probability)) + "%", (left, bottom + 12), font, 0.5, (255, 255, 255), 1)

    #     # save every frames with detected face to image ---> Do not use in production
    #     cv2.imwrite('./ati_frames/frame' + str(frame_number) + '.jpg', frame)
    # else:
    #     log('No image in frame ' + str(frame_number))
    #     # save every frames without detected face to image ---> Do not use in production
    #     cv2.imwrite('./ati_frames/no_face/frame' + str(frame_number) + '.jpg', frame)

    # display frame
    cv2.imshow('Video', frame) 

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Do a bit of cleanup
log('Number of frames = ' + str(frame_number))
log('Exiting Program and cleanup stuff')
logfile.close()
video_capture.release()
cv2.destroyAllWindows()
