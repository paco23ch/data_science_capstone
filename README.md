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
  - Normalizing: We also need to normalize the data when creating the tensors, so given that each of the values in the (224,244,3) vector are  in the range of 0-255, we will divide them all by 255 to normalize in the scale 0-1 for the training to work correctly.

## 3.2. Implementation
### 3.2.1. Face detection
This function was created to human faces, using the OpenCV.  For this we also need to transform the images, since OpenCV identifies faces on a grey scale image. This function basically returns a True/False depending on a face being present.  One of the main limitations of the OpenCV algorithms is that faces need to be frontal, this could be resolved by training a CNN with faces in different orientations, but given the time, this was not possible.

As mentioned before, this function was to be tested against both dog and human images to determine the performance, and this is how the function works:
- Faces detected 100/100 - this is an accuracy of 100% in face detection.
- Dogs detected 11/100 - this is an accuracy of 11% in dog detection.  The desired output here was to get a 0%, becasue none of these images are dogs, but when exploring the images we noticed that some images do contain humans in them, so that could be the reason for some of these 11% detected human faces.
### 3.2.2. Dog detection
This function was created to identify dogs in an image, based on the ResNet50 library, which has been trained to identify all sorts of images from a long library.  When using ResNet50, we need to consider a given range for dogs to be detected.  So if a label from ResNet50 is between 151 and 268, it means there's a dog in the image.

Again, we ran both the human and dog images through the algorithm to determine how accurate the dog detector is and these are the results:
- Faces detected 0/100, which is what was expected, no dogs in human pictures.
- Dog faces detected 100/100, which is a great result, because we know all dog images received actually have a dog, even though they may also contain a human in them in some cases.
### 3.2.3. Identifying dog breeds
#### 3.2.3.1. From sratch
The first attempt at actually training a network is to build a network from scratch, with a combination of dense and convolutional networks, which as demonstrated in the notebook, can take many epochs and processing time.  As can be seen on the notebook, a 3.5% precision for a from-scratch network was obtained in 8 epochs.  

The architecture that was selected is a combination of Convolutional network and Max Pooling layers, which help reduce the dimensionality of the image.  Each convolutional network reduces the 'image' size by half (224->112->56->28) and at the same time it extends the depth by using each of the filters, doubling them every time (16,32,64).  At the end of the pipeline, we have a Max Pooling layer, which will further reduce the dimensionality of the image, and at the end we'll have a Dense layer, which will have 133 outputs, just as the labels for each of the dog breeds.
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 224, 224, 16)      208       
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 112, 112, 16)      0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 112, 112, 32)      2080      
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 56, 56, 32)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 56, 56, 64)        8256      
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 28, 28, 64)        0         
_________________________________________________________________
global_average_pooling2d_1 ( (None, 64)                0         
_________________________________________________________________
dense_1 (Dense)              (None, 133)               8645      
=================================================================
Total params: 19,189
Trainable params: 19,189
Non-trainable params: 0
_________________________________________________________________
```

This is a long process to train a network from sratch, as can be seen, we could only obtain a 10.28% accuracy on the test set after 100 epochs lasting 35 minutes.  At the beginning of the training process, the model makes continuous improvements, but as epochs progress, validation loss doesn't improve as often.   All in all, Validation loss increased from 4.8682 to 4.1385 and accuracy increased from 0.0120 to 0.1042, which suggests that training this network for multiple hours could increase it's accuracy.

#### 3.2.3.2. From a pre-trained model
Instead of using a network from scratch, we will transfer learning from a pre-trained network and add a classifier at the end.  In order to shorten the training period, and reuse already trained networks, we'll build a two step network, sort of pipeline.
- __Extracting features__: We'll use the multiple alternatives available in the keras package to extract the features of the images
- __Classifying features__:  Once a set of features are identified, we'll use another set of layers to correctly classify each of the breeds
Also, to save time, we were given a series for NPZ files which contain the feature extraction for the train, test and validation sets for each of the keras pre-trained networks.  

Each of these networks have a different output, in some cases it's a vector of (1,1,2048) dimensions and in others (7,7,2048), so for tuning purposes we used the shape of the vectors given to save time, but in the end, the selected pre-trained network output shape had to be checked and explicitly typed into the code for it to work.

Given that we want to reduce the dimensionality of the pre-trained network further, the selected architecture for the feature classifier is as follows.  This network contains a Global Pooling average layer and then a Dense Layer with 133 outputs, one for each breed.
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
global_average_pooling2d_14  (None, 2048)              0         
_________________________________________________________________
dense_14 (Dense)             (None, 133)               272517    
=================================================================
Total params: 272,517
Trainable params: 272,517
Non-trainable params: 0
_________________________________________________________________
```
### 3.2.4.  Web application
The web application is based on code from a previous Udacity project on Disaster message classification.  It's a Flask-based application that uses Jinja to generate HTML code via templates.   

The code is split in 3 main python files and 3 html files.   The main python file `run.py` contains the Flask skeleton and creates a `/` and `index` path to handle the first time a uses logs into the web page by loading the `master.html` template.  The `run.py`file instantiates a class contained in the `dog_recognition.py` file, that pre-loads the best model and weights obtained in the training process.

The app has a very simple interface, that will allow you to select a file using a file chooser and once you submit your uploaded image, which will invoke the path `/go`, which will in turn use the `DogClassifier` instance to run the uploaded image through a couple of functions `dog_detector` and `face_detector` to determine if the image contains a dog or a person, and will run the image through the pre-trained/classifer pipeline to gather the predicted dog breed for the image.
Using the values from `dog_detector` and `face_detector` which are True or False, the module will render either the `go.html` template, in case a dog or a person was found or the `error.html` template in case none were found.

When a person was identified, the app will acknowledge and let them know that the image contains features of a given breed.  For dog breeds it will give it's prediction.  In both cases, besides showing the certainty of the prediction, it will also show sample images of that dog breed so the user can compare the results and understand why that prediction was made.  There are cases where multiple dog breeds may look alike, and in the case of humans, there must've been a reason why that given breed was selected.
## 3.5. Refinement
The first refinement done on the model was on the Classifying features section of the pipeline.  The first step of the network will always be the Global Average Pooling network, plus one or more Dense networks to generate the final classifier.   However, in the end, after multiple tests, we would tell that adding more than one layer of classifiers actually made accuracy worse than with a single layer, so I decided for a single Global Average Pooling layer, plus a Dense layer at the end.

The other part in the pipeline is the feature identification first stage, using multiple options in the `keras` package: `ResNet50`, `VGG16`, `VGG19`, `Inception`, `Xception`.  Fortunately for us, on the training stage, the feature extraction for training from those models was given to us, so all we needed to do in the actual application is to load the `keras` library and use the values, however, for the application at runtime, we did have to load the model from the selected package and run each of the input images through it.

# 4. Results
## 4.1. Model Evaluation and Validation
In order to look at different pre-trained networks, I ran 100 epochs for each of the ones listed on the table, and the best was the Xception network, so I decided to use it as the pretrained network to get the highest possible accuracy.  ResNet50 at some training iteration even provided as high as 82% accuracy, but Xception was the highest in the end.

In this case, the model was trained on the train set, validated through the epochs against validation set and tested against the test set in one shot at the end.  Only dog data was used to test the model.  

In order to compare models through the same parameters, I decided to keep the epochs at 100, but based on the Loss ans accuracy graph for the Xception network, we can see that training our Classifier is really fast, we really only need a few (<5) epochs for these networks to train:

| Loss | Accuracy |
| :-- | :--: |
| !()[./Loss.png] | !()[./Accuracy.png] |

The final accuracy against the test data can be seen here:

| Model | Test accuracy |
| :-- | :--: |
| ResNet50 | 78.3493% |
| DogVGG16Data | 49.7608% |
| DogVGG19Data | 49.1627% |
| DogInceptionV3Data | 79.0670% | 
| DogXceptionData | 84.6890% |
## 4.2. Justification
Given that we're looking for the highest accuracy possible, and the algorithm has been validated on each epoch for the loss, and also has been tested for accuracy with the test data set and we've obtained close to 85% it looks like the right model to use.  After testing different combinations of the pre-trained networks and classifier combinations, the best performance was obtained with the Xception package.

In a previous classifier I wrote, VGG19 performed well in sklearn, but in this case VGG16 and VGG19 both underperformed to the rest.  Although the performance between ResNet50, Inception and Xception are close to each other, I selected the Xception moduls which gave the highest test accuracy.

# 5. Conclusion
# 5.1. Reflection
When I built the first flower classifier and some of the recommendation engine exercises I always wondered how would this be applied in real life, for instance I'm about to work on a project that should give users recommendations on similar web pages as they are located in.   So, for me the disaster message analysis project and this one was ver insightful on how to accomplish this in real life.

It was very interesting to take a set of images, and first build very basic detectors of a person and a dog, based on existing libraries, and then work on the alternatives of a CNN.  Trying to build one from scratch could be the best way to tailor it as much as you want, however, that takes a lot of time, resources and as mentioned across the class videos, experience.  This is a combination of art and science, which takes time to master.  Given that you can use pre-trained networks should make our lifes easier, so it's very interesting to experiment with each of those alternatives to achieve the best results possible.

And then connecting all the pieces together with the web application was the greatest part of all.  I was able to setup this very simple interface, which could definitely be improved, and using the model in the background to run the images through it and obtain an actual prediction.

A couple of aspects that I found interesting were:
- As I tried to play with the pre-trained network options, it was suprising that the VGG models didn't perform as well as the others.  Like I mentioned before, I used that architecture in the flower classification project and work very well, so I wonder if either the model has to be trained even more for these networks to work or maybe each architecture works best for some kinds of images.   I'm more inclined to the latter because as we used ResNet50 to identify dogs, it has been trained for other images as well, so I guess with enough time and resources any network could perform as well as the other.  Plus, there's always the classifier itself, which could also be tweaked to perform better.  Like mentioned before, combination of art and science.
- One of the drawbacks of using CNNs for other purposes is not so much on the accuracy they can have, but rather the amount of resources and/or time it's required to trained them.  In this project training time wasn't really an issue because we had a GPU enabled workspace, but I think going forward I need to find a suitable option for myself to be able to train networks for future projects.   I did try at some point in the Nanodegress to use my local machine without a GPU and it's definitely not an ideal scenario.
# 5.2. Improvement
In general the model was able to produce the right results, however, there are a few things could be done to improve the precision.
- We could add image shifting and rotating to the existing images to have more variability of features.  The variability would allow the networks to identify the features in different combinations and places.
- We could also add more images to the training set to add the samples, and also use Dropout layers.  The Dropout layers should allow for a better training of the network, so the network has input variability as well.
- We could try other available pre-trained networks (https://keras.io/api/applications/).  Keras has multiple versions of ResNet available at the previous link, as well as other Inception entworks.  Also a few other that could prodice different results.
- Using existing trained networks will save time, but we could also propose a new architecture end-to-end and train for a few more epochs. This would take more time and resources as shown in one of the earlier sections.

# 6. Deliverables
## 6.1. Notable exclusions
In order to minimize the size of the repository I excluded the training, testing and validating images.  I did include a set of images (test) that will work with the application to show users similar dogs to the ones identified, specially in the case of people, so they try to find the similarl identified features.
## 6.2. Repository contents
The code contained here is split basically in a working notebook, in order to train the network and refine/improve the model, plus build the general algorithm, plus a web application.
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
## 6.3. Web application
In order to run locally, you just need to type `python run.py` in the app directory, and the app will load an http server, accesible on `http://localhost:3001`.   
There may be cases where because of overlapping features, the certainty of the prediction is lower or it even predicts a different breed, but you will be able to see why the algorithm may be confusing the breed.

