import numpy as np
from numpy import expand_dims
from numpy import load
from numpy import asarray
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from keras.models import load_model
from os import listdir
from os.path import isdir
from PIL import Image
from mtcnn.mtcnn import MTCNN
from datetime import datetime
from time import sleep
from time import time
import cv2
import argparse

welcome = 'Face recognition application'

parser = argparse.ArgumentParser(description=welcome)
parser.add_argument('--train_data', default='ati_employee_single/train/', help='data images for trainning')
parser.add_argument('--emb_model_path', default='facenet_keras.h5', help='model used to encode face image')
parser.add_argument('--test_video', default='video_test.avi', help='video used for testing, to get from camera just input 0')
parser.add_argument('--recognition_threshold', default=1.0, help='threshold distance to recgonize faces')
parser.add_argument('--output_folder', default='ati_frames/', help='where to save each frame to image')
args = parser.parse_args()
train_data = args.train_data
emb_model_path = args.emb_model_path
test_video = args.test_video
recognition_threshold = args.recognition_threshold
output_folder = args.output_folder

# extract a single face from a given photograph
def extract_face(filename, face_detector, required_size=(160, 160)):
    # load image from file
    image = Image.open(filename)
    # convert to RGB, if needed
    image = image.convert('RGB')
    # convert to array
    pixels = asarray(image)
    # create the detector, using default weights
    # detector = MTCNN()
    # detect faces in the image
    results = face_detector.detect_faces(pixels)
    # extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']
    # bug fix
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array
 
# load images and extract faces for all images in a directory
def load_faces(directory, face_detector):
    faces = list()
    # enumerate files
    for filename in listdir(directory):
        # path
        path = directory + filename
        # get face
        face = extract_face(path, face_detector)
        # store
        faces.append(face)
    return faces
 
# load a dataset that contains one subdir for each class that in turn contains images
def load_dataset(directory, face_detector):
    X, y = list(), list()
    # enumerate folders, on per class
    for subdir in listdir(directory):
        # path
        path = directory + subdir + '/'
        # skip any files that might be in the dir
        if not isdir(path):
            continue
        # load all faces in the subdirectory
        faces = load_faces(path, face_detector)
        # create labels
        labels = [subdir for _ in range(len(faces))]
        # summarize progress
        print('>loaded %d examples for class: %s' % (len(faces), subdir))
        # store
        X.extend(faces)
        y.extend(labels)
    return asarray(X), asarray(y)
 
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
    # norm_yhat = in_encoder.transform(yhat[0])
    return yhat[0]

# create the detector, using default weights
face_detector = MTCNN()
print('Face detector created')

# load train dataset
trainX, known_names = load_dataset(train_data, face_detector)
# normalize input vectors
in_encoder = Normalizer(norm='l2')
font = cv2.FONT_HERSHEY_DUPLEX

logfile = open('logs/' + str(datetime.now()) + '.log', 'w')
def log(msg):
    print(f'\n [INFO] {msg}', file=logfile)
    print('\n [INFO] ' + msg)
    return 0

# load the facenet model
emb_model = load_model(emb_model_path)
print('Loaded Model')

# convert each face in the train set to an embedding
known_face_encodings = list()
for face_pixels in trainX:
    embedding = get_embedding(emb_model, face_pixels)
    known_face_encodings.append(embedding)
known_face_encodings = asarray(known_face_encodings)
print(known_face_encodings.shape)

known_face_encodings = in_encoder.transform(known_face_encodings)
print(known_face_encodings)


def face_distance(face_encodings, face_to_compare):
    if len(face_encodings) == 0:
        return np.empty((0))
    return np.linalg.norm(face_encodings - face_to_compare, axis=1)
    # distances = list()
    # for vec in face_encodings:
        # distances.append(cosine(vec, face_to_compare))
    # return distances

for vec, name in zip(known_face_encodings, known_names):
    res = face_distance(known_face_encodings, vec)
    print(name)
    print(res)

video_capture = cv2.VideoCapture(test_video)

required_size=(160, 160)
color = (0, 0, 255)
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

    pixels = asarray(frame)
    results = face_detector.detect_faces(pixels)
    if (len(results) > 0):
        for face in results:
            tic = time()
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
            # compute Euclidean distance between faces in each frame with all known faces
            face_encoding = get_embedding(emb_model, face_image)
            face_encoding = expand_dims(face_encoding, axis=0)
            face_encoding = in_encoder.transform(face_encoding)
            distances = face_distance(known_face_encodings, face_encoding)
            recognized_name = 'Unknown'
            # find the closest face in known faces
            closest = np.argmin(distances)
            distance = str(distances[closest])
            if (distances[closest] < recognition_threshold):
                recognized_name = known_names[closest]
            # display result on frame
            cv2.rectangle(frame, (left,top), (right,bottom), color, 2)
            cv2.rectangle(frame, (left - 1,top), (right + 1,top - 20), color, cv2.FILLED)
            cv2.rectangle(frame, (left - 1,bottom), (right + 1,bottom + 20), color, cv2.FILLED)
            cv2.putText(frame, recognized_name, (left, top - 6), font, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, distance[:5], (left, bottom + 12), font, 0.5, (255, 255, 255), 1)
            toc = time()
            log('In frame ' + str(frame_number) + ', recognized ' + recognized_name + ' with distance ' + str(distance) + ' after ' + str(toc - tic) + ' second')
        # save every frames with detected face to image ---> Do not use in production
        cv2.imwrite(output_folder + 'frame' + str(frame_number) + '.jpg', frame)
    else:
        log('No image in frame ' + str(frame_number))
        # save every frames without detected face to image ---> Do not use in production
        cv2.imwrite(output_folder + 'no_face/frame' + str(frame_number) + '.jpg', frame)
    cv2.imshow('Video', frame) 

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Do a bit of cleanup
log('Number of frames = ' + str(frame_number))
log('Exiting Program and cleanup stuff')
logfile.close()
video_capture.release()
cv2.destroyAllWindows()
