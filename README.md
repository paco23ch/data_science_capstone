# Udacity Data Scientist Capstone Project
__Dog Breed Detection__
*Francisco Chavez Clemente*

# 1. Project Definition
## 1.1. Project Overview
One of the most popular pets in the world are dogs.  There are a large number of breeds and for all likings, and for someone who is not familiar with all breeds it may be difficult to identify the correct one for any given dog.  At the same time, it could be a fun app to receive images of people and trying to tell them which dog breed they may look like given the identified features in the image.

The input data for this project are images of different sizes.  We have dog images and people images.  The dog images are split in three sets train, validation and testing. 

## 1.2. Problem Statement
We need to be able to take an input image and using a model identify the race with a test/validation precision of 80% at least

## 1.3. Metrics
- We will use accuracy as the reference metric.  We will use the predictions vs. the actual labels on the train, test and validation sets to determine the effectiveness of the algorithm.
- To do an early stop we will measure the loss change from iteration to iteration when running the model against the validation data set.  As optimizer we'll use  `rmsprop` and for loss we'll use `categorical_crossentropy`

# 2. Analysis
## 2.1. Data Exploration & Visualization
In order for the algorithms to work, we will need to pre-process the input images to the right size.

# 3. Methodology
## 3.1. Data Preprocessing
In order to use the data we need to transform to the a standard size (224,224,3), because they all are color images.  Plus, the algorithms will need to use tensors to pass the images through them.

## 3.2. Implementation
### 3.2.1. Identifying a user
We'll need to create a function to identify human faces, using the OpenCV.  For this we will also need to transform the images, since OpenCV identifies faces on a grey scale image.  This function will basically return a True/False depending on a face being present.  One of the main limitations of the OpenCV algorithms is that faces need to be frontal, this could be resolved by training a CNN with faces in different orientations, but given the time, this was not possible.

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
