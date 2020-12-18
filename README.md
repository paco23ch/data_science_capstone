# Udacity Data Scientist Capstone Project
__Dog Breed Detection__
*Francisco Chavez Clemente*

# 1. Project Definition
## 1.1. Project Overview
One of the most popular pets in the world are dogs.  There are a large number of breeds and for all likings, and for someone who is not familiar with all breeds it may be difficult to identify the correct one for any given dog.  At the same time, it could be a fun app to receive images of people and trying to tell them which dog breed they may look like given the identified features in the image.

The input data for this project are images of different sizes.  We have dog images and people images.  The dog images are split in three sets train, validation and testing. 

## 1.2. Problem Statement
We need to be able to take an input image and using a model identify the race with a test/validation precision of 80% at least.   The application should be able to tell if there is a person in the picture and let them know the similarity to a given dog breed and the potencial feature overlaps.   If it's a dog, it should tell the user the breed and the certainty with which the model was able to predict.

In order to fulfill the above, we needed to build:
- Human detector: A component that can detect faces in an image, for which we used the OpenCV library
- Dog detector: A component that can quickly tell if there's a dog in the image, for which we used one of the keras libraries, such as ResNet50, that has been train in identifying a number of different images
- Dog classifier: A component that can determine which specific breed a dog is, based on the input images for each of the 133 breeds.  For this component, we used one of the pretrained networks and add a classifier for the specific number of breeds we want.  This will allow us to save time in training and also to train based on our own images.  This component will require a number of epochs and tuning in order to make it have accuracy above 60% as required.
- Application: A component that can interact with the user to receive an image and return a result.  For this, we built a Flask application that can receive files via a web page and use the models developed above to predict the image contents.

## 1.3. Metrics
- Human detector. Since the main purpose of this component is to know if the image contains a human face, we will use:
  - Accuracy = Number of correct images / Total images
  - We will apply this metric to both the dog and human training sets in order to determine their precision.
- Dog detector. Since the main purpose of this component is to know if the image contains a dog, we will use:
  - Accuracy = Number of correct images / Total images
  - We will apply this metric to both the dog and human training sets in order to determine their precision.
- Dog classifier. This will require a couple of metrics used in the training process, as well as one in the test process:  
  - Training:
    - Loss = sum(Correct value - Predicted value)/Total values 
    - Accuracy = Number of correctly predicted images / Total Images 
    - We will use both metrics during training process on the train & validation sets to:
      - Adjust the network parameters
      - Determine when the model has improved or not
  - Testing
    - Accuracy = Number of correctly predicted images / Total Images 
    - We will use the predictions vs. the actual labels on the test set to determine the effectiveness of the algorithm once it has been trained.

# 2. Analysis
## 2.1. Data Exploration 
We have two sets of data files: faces and dogs.
- Faces: This is a sample directory, which contains a directory per person, and each directory contains one or more images to be used as reference for the face detector component.
- Dogs: There are three data sets: train, test and validation.   Each of these directories contain 133 directories with images representing each of the breeds that we're trying to identify.   
  - The directories are named as 999.Dob_breed_name, where 999 goes from 001 to 133.  From the directory names, we should be able to obtain the image labels and names, for example, directory `001.Affenpinscher` will be label 1, and the breed name is __Affenpinscher__.  We should be able to use each of the images sorting these directories and knowing from it the correct target label.
  - As for the images, by exploring them, we can see that there are all sorts of resolution, which could be an issue if we use them as-is to train our network.  We will most likely have to resize them to a standard window in order to detect features.
  - In some cases, the images do contain people with the dogs, which will possibly lead to us detecting human faces and mistakenly saying there's a person in the picture.

In order for the algorithms to work, we will need to pre-process the input images to the right size.
## 2.2. Visualization
It's hard to find a set of visualizations for the input data, being a set of images, but as can be seen in the notebook, the OpenCV library is able to correctly identify the faces of an image, such as images show in each of the steps.

What could also constitute a type of metric/visualization are the basic stats about the data set:
- Faces
  - There are 13233 total human images.
- Dogs
  - There are 133 total dog categories.
  - There are 8351 total dog images.
  - There are 6680 training dog images.
  - There are 835 validation dog images.
  - There are 836 test dog images.

# 3. Methodology
## 3.1. Data Preprocessing
In order to use the data, we will need to do a couple of steps for the training stages of the breed recognition algorithms, given that each of the functions require specific structures.
  - Re-sizing: In order to use the dog data we need to transform to the a standard size (224,224,3), because they all are color images, so this means the image size is 224x224, and there are 3 channels of data (RGB).  
  - Vectorizing: The algorithms will need to use tensors to pass the images through them, so we will also convert the resized image to a vector, using the image libraries from keras.preprocessing.

## 3.2. Implementation
### 3.2.1. Face detection
We created a function to identify human faces, using the OpenCV.  For this we also need to transform the images, since OpenCV identifies faces on a grey scale image.  This function basically returns a True/False depending on a face being present.  One of the main limitations of the OpenCV algorithms is that faces need to be frontal, this could be resolved by training a CNN with faces in different orientations, but given the time, this was not possible.

As mentioned before, this function was to be tested against both dog and human images to determine the performance, and this is how the function works:
- Faces detected 100/100 - this is an accuracy of 100% in face detection.
- Dogs detected 11/100 - this is an accuracy of 11% in dog detection.  The desired output here was to get a 0%, becasue none of these images are dogs, but when exploring the images we noticed that some images do contain humans in them, so that could be the reason for some of these 11% detected human faces.
### 3.2.2. Dog detection
We created a function to identify dogs in an image, based on the ResNet50 library, which has been trained to identify all sorts of images from a long library.  When using ResNet50, we need to consider a given range for dogs to be detected.  So if a label from ResNet50 is between 151 and 268, it means there's a dog in the image.

Again, we ran both the human and dog images through the algorithm to determine how accurate the dog detector is and these are the results:
- Faces detected 0/100, which is what was expected, no dogs in human pictures.
- Dog faces detected 100/100, which is a great result, because we know all dog images received actually have a dog, even though they may also contain a human in them in some cases.
### 3.2.2. Identifying dog breeds
#### 3.2.2.1. From sratch
The first attempt at actually training a network is to build a network from scratch, with a combination of dense and convolutional networks, which as demonstrated in the notebook, can take many epochs and processing time.  As can be seen on the notebook, a 3.5% precision for a from-scratch network was obtained in 8 epochs.  

This could be a long process to train a network from sratch, as can be seen, we could only obtain a 10.28% accuracy on the test set after 100 epochs lasting 35 minutes.

#### 3.2.2.2. From a pre-trained model
So, we'll transfer knowledge from a pre-trained network to a new one and add a classifier at the end. In order to shorten the training period, and reuse already trained networks, we'll build a two step network, sort of pipeline.
- __Extracting features__: We'll use the multiple alternatives available in the keras package to extract the features of the images
- __Classifying features__:  Once a set of features are identified, we'll use another set of layers to correctly classify each of the breeds

## 3.3. Refinement
The first refinement done on the model was on the Classifying features section of the pipeline.  The first step of the network will always be the Global Average Pooling network, plus one or more Dense networks to generate the final classifier.   However, in the end, after multiple tests, we would tell that adding more than one layer of classifiers actually made accuracy worse than with a single layer, so I decided for a single Global Average Pooling layer, plus a Dense layer at the end.

Another moving part in the pipeline is the feature identification first stage, using multiple options in the `keras` package: `ResNet50`, `VGG16`, `VGG19`, `Inception`, `Xception`.  Fortunately for us, on the training stage, the feature extraction for training from those models was given to us, so all we needed to do in the actual application is to load the `keras` library.
# 4. Results
## 4.1. Model Evaluation and Validation
In order to look at different pre-trained networks, I ran 100 epochs for each of the ones listed on the table, and the best was the Xception network, so I decided to use it as the pretrained network to get the highest possible accuracy.  ResNet50 at some training iteration even provided as high as 82% accuracy, but Xception was the highest in the end.

| Model | Test accuracy |
| :-- | :--: |
| ResNet50 | 78.3493% |
| DogVGG16Data | 49.7608% |
| DogVGG19Data | 49.1627% |
| DogInceptionV3Data | 79.0670% | 
| DogXceptionData | 84.6890% |
## 4.2. Justification
Given that we're looking for the highest accuracy possible, and the algorithm has been validated on each epoch for the loss, and also has been tested for accuracy with the test data set and we've obtained close to 85% it looks like the right model to use.
# 5. Conclusion
# 5.1. Reflection
With CNNs, we can very nicely simplify the input images to smaller vectors, which in turn can be simplified further to come to a more simple set of features, for a later classificator implementation.   This approach saves lots of time on training, specially when those have been trained with lots of images to identify features, which are helpful to focus on the classification stage.
# 5.2. Improvement
In general the model was able to produce the right results, however, there are a few things could be done to improve the precision.
- We could add image shifting and rotating to the existing images to have more variability of features.
- We could also add more images to the training set to add the samples, and also use Dropout layers.
- We could try other available pre-trained networks.
- Using existing trained networks will save time, but we could also propose a new architecture end-to-end and train for a few more epochs. This would take more time and resources as shown in one of the earlier sections.
# 6. Deliverables
## 6.1. Notable exclusions
In order to minimize the size of the repository I excluded the training, testing and validating images.  I did include a set of images (test) that will work with the application to show users similar dogs to the ones identified, specially in the case of people, so they try to find the similarl identified features.
## 6.2. Repository contents
The code contained here is split basically in a working notebook, in order to train the network and refine/improve the model, plus build the general algorithm, plus a web application.
## 6.3. Web application
The web application is based on code from a previous Udacity project on Disaster message classification.  It's a Flask-based application that uses Jinja to generate HTML code via templates.   It has a very simple interface, that will allow you to select a file using a file chooser and once you submit your uploaded image, it will give you the dog breed prediction along with sample images to help identify races.  

In order to run locally, you just need to type `python run.py` in the app directory, and the app will load an http server, accesible on `http://localhost:3001`.   

There may be cases where because of overlapping features, the certainty of the prediction is lower or it even predicts a different breed, but you will be able to see why the algorithm may be confusing the breed.

### 6.3.1. Directory Structure

- __app__: Main app directory
  - __model__: Directory where the model and face features config file is stored
  - __static__: Directory for image upload and reference
    - __reference__: Contains directories for sample images for all dog breeds
    - __upload__: Where files posted to the app are stored
  - __templates__: This contains the HTML Jinja templates used
    - `go.html`: Will pocess the results from each of the evaluation requests
    - `master.html`: Main page that will prepare the input fields and submit buttons
  - `dog_recognition.py`: Contains the DogRecognition class which encapsulates the code for the predictive features in the backend
  - `extract_bottleneck_features.py`: Used to extract the correct features for the pre-trained network, contains each of the 5 used in this notebook
  - `run.py`: Main application file
- __haarcascades__: Contains the face detection XML configuration file
- __images__: Sample images to test the application
- __requirements__: List of code pre-reqs.
- __saved_models__: Models produced while running the notebook.
- `README.md`: This file
- `dog_app.html`: HTML output of the notebook used for training and exploring the different network architectures
- `dog_app.ipynb`: Jupyter Notebook used for training and exploring the different network architectures
- `extract_bottleneck_features.py`: Used to extract the correct features for the pre-trained network, contains each of the 5 used in this notebook
