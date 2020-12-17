import cv2     
import numpy as np
from keras.preprocessing import image                  
#from keras.applications.resnet50 import preprocess_input, decode_predictions, ResNet50
from keras.applications.xception import preprocess_input, decode_predictions, Xception
from keras.layers import GlobalAveragePooling2D, Dense
from keras.models import Sequential
from extract_bottleneck_features import extract_Resnet50, extract_Xception
from glob import glob

class DogRecognizer:
    """
    DogRecognizer helps to initialize a single object with the models and be able to run multiple times
    It will also keep some utility functions
    """

    def __init__(self):
        """
        Class initializer

        Args:
        self - The current object

        Returns:
        None
        """

        # Instantiate the model as a sequential keras model and add the layers
        self.my_model = Sequential()
        # self.my_model.add(GlobalAveragePooling2D(input_shape=(1, 1, 2048))) # For ResNet50
        self.my_model.add(GlobalAveragePooling2D(input_shape=(7, 7, 2048))) # For XCeption
        self.my_model.add(Dense(133, activation='softmax'))

        # Load the best model
        self.my_model.load_weights('model/weights.best.mymodel.hdf5')

        # extract pre-trained face detector
        self.face_cascade = cv2.CascadeClassifier('model/haarcascade_frontalface_alt.xml')

        # Initialize the dog names array and the dog directories to be used as sample images
        self.dog_names = [item[21:-1] for item in sorted(glob("static/reference/*/"))]
        self.dog_directories = [item[:] for item in sorted(glob("static/reference/*/"))]

        return

    # returns "True" if face is detected in image stored at img_path
    def face_detector(self, img_path):
        """
        Function to detect if there's a face in the image

        Args:
        Path of the image where we want to detect the faces

        Returns:
        True/False if there's a face or not
        """

        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray)
        return len(faces) > 0

    def path_to_tensor(self, img_path):
        """
        Load and convert the image to a tensor to run through the models

        Args:
        Path of the image we want to convert to a tensor

        Returns:
        numpy array containing the loaded image
        """

        # loads RGB image as PIL.Image.Image type
        img = image.load_img(img_path, target_size=(224, 224))
        # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
        x = image.img_to_array(img)
        # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
        return np.expand_dims(x, axis=0)

    def mymodel_predict_breed(self, img_path):
        """
        Predict the dog breed using the model

        Args:
        Path of the image we want to predict

        Returns:
        Index of the highest probability prediction found
        Highest probability prediction found
        """

        #bottleneck_feature = extract_Resnet50(self.path_to_tensor(img_path))
        bottleneck_feature = extract_Xception(self.path_to_tensor(img_path))

        predicted_vector = self.my_model.predict(bottleneck_feature)

        return np.argmax(predicted_vector), predicted_vector[0,np.argmax(predicted_vector)]


    def predict_breed(self, img_path):
        """
        This is the main method to be called, which will also identify if there was a human in the picture and will return some samples images
        besides the actual prediction name and probability

        Args:
        Path of the image we want to convert to a tensor

        Returns:
        numpy array containing the loaded image
        """

        # Find out if there's a face in the image
        is_human = False
        if self.face_detector(img_path):
            is_human = True
                
        # Call the prediction function and find the name and sample images to return
        pred_index, pred_prob = self.mymodel_predict_breed(img_path)
        pred_name = self.dog_names[pred_index].replace('_',' ').title()
        sample_images = [sorted(glob(self.dog_directories[pred_index] + "*"))]
        
        return (pred_name, pred_prob, is_human, sample_images)