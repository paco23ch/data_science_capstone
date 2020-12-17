import cv2     
import numpy as np
from keras.preprocessing import image                  
from keras.applications.resnet50 import preprocess_input, decode_predictions, ResNet50
from keras.layers import GlobalAveragePooling2D, Dense
from keras.models import Sequential
from extract_bottleneck_features import extract_Resnet50
from glob import glob

class DogRecognizer:

    def __init__(self):
        self.my_model = Sequential()
        self.my_model.add(GlobalAveragePooling2D(input_shape=(1, 1, 2048)))
        self.my_model.add(Dense(133, activation='softmax'))
        self.my_model.load_weights('model/weights.best.mymodel.hdf5')

        # Formula for obtaining the dog names:  
        self.dog_names = [item[21:-1] for item in sorted(glob("static/reference/*/"))]
        self.dog_directories = [item[:] for item in sorted(glob("static/reference/*/"))]

        print(self.dog_directories[0])

        return

    # returns "True" if face is detected in image stored at img_path
    def face_detector(self, img_path):
        # extract pre-trained face detector
        face_cascade = cv2.CascadeClassifier('model/haarcascade_frontalface_alt.xml')

        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray)
        return len(faces) > 0

    def path_to_tensor(self, img_path):
        # loads RGB image as PIL.Image.Image type
        img = image.load_img(img_path, target_size=(224, 224))
        # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
        x = image.img_to_array(img)
        # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
        return np.expand_dims(x, axis=0)

    def mymodel_predict_breed(self, img_path):
        # extract bottleneck features
        bottleneck_feature = extract_Resnet50(self.path_to_tensor(img_path)) #.astype('float32')/255)
        # obtain predicted vector
        predicted_vector = self.my_model.predict(bottleneck_feature)
        # return dog breed that is predicted by the model
        return np.argmax(predicted_vector), predicted_vector[0,np.argmax(predicted_vector)]
        #return np.round(predicted_vector*100,0).astype(int)

    def predict_breed(self, img_path):
        is_human = False

        if self.face_detector(img_path):
            is_human = True
                
        pred_index, pred_prob = self.mymodel_predict_breed(img_path)
        pred_name = self.dog_names[pred_index].replace('_',' ').title()
        sample_images = [sorted(glob(self.dog_directories[pred_index] + "*"))]
        
        return (pred_name, pred_prob, is_human, sample_images)