# Udacity Data Scientist Capstone Project
> __Dog Breed Detection__

> *Francisco Chavez Clemente*

# Project Definition
## Project Overview
One of the most popular pets in the world are dogs.  There are a large number of breeds and for all likings, and for someone who is not familiar with all breeds it may be difficult to identify the correct one for any given dog.  At the same time, it could be a fun app to receive images of people and trying to tell them which dog breed they may look like given the identified features in the image.

## Problem Statement
We need to be able to take an input image and using a model identify the race with a test/validation precision of 80% at least

## Metrics
- We will use accuracy as the reference metric.  We will use the predictions vs. the actual labels on the train, test and validation sets to determine the effectiveness of the algorithm.
- To do an early stop we will measure the loss change from iteration to iteration when running the model against the validation data set.  As optimizer we'll use  `rmsprop` and for loss we'll use `categorical_crossentropy`

# Analysis
## Data Exploration & Visualization
The input data for this project are images of different sizes.  We have dog images and people images.  The dog images are split in three sets train, validation and testing.  In order for the algorithms to work, we will need to pre-process these images.

# Methodology
## Data Preprocessing
In order to use the data we need to transform to the a standard size (224,224,3), because they all are color images.  Plus, the algorithms will need to use tensors to pass the images through them.

## Implementation
### Identifying a user
We'll need to create a function to identify human faces, using the OpenCV.  For this we will also need to transform the images, since OpenCV identifies faces on a grey scale image.  This function will basically return a True/False depending on a face being present.

### Identifying dog breeds
The first attempt at actually training a network is to build a network from scratch, with a combination of dense and convolutional networks, which as demonstrated in the notebook, can take many epochs and processing time.  So we'll transfer knowledge from a pre-trained network to a new one and add a classifier at the end.

In order to shorten the training period, and reuse already trained networks, we'll build a two step network, sort of pipeline.
- __Extracting features__: We'll use the multiple alternatives available in the keras package to extract the features of the images
- __Classifying features__:  Once a set of features are identified, we'll use another set of layers to correctly classify each of the breeds

## Refinement
The first refinement donde on the model was on the Classifying features section of the pipeline.  The first step of the network will always be the Global Average Pooling network, plus one or more Dense networks to generate the final classifier.   However, in the end, after multiple tests, we would tell that adding more than one layer of classifiers actually made accuracy worse than with a single layer, so I decided for a single Global Average Pooling layer, plus a Dense layer at the end.

## Improvement
In order to look at different pre-trained networks, I ran 100 epochs for each of the ones listed on the table, and the best was the Xception network, so I decided to use it as the pretrained network to get the highest possible accuracy.  ResNet50 at some training iteration even provided as high as 82% accuracy, but Xception was the highest in the end.

| Model | Test accuracy |
| :-- | :--: |
| ResNet50 | 78.3493% |
| DogVGG16Data | 49.7608% |
| DogVGG19Data | 49.1627% |
| DogInceptionV3Data | 79.0670% | 
| DogXceptionData | 84.6890% |

# Deliverables
## Notable exclusions
In order to minimize the size of the repository I excluded the training, testing and validating images.  I did include a set of images (test) that will work with the application to show users similar dogs to the ones identified, specially in the case of people, so they try to find the similarl identified features.
## Application
The code contained here is split basically in a working notebook, in order to train the network and refine/improve the model, plus build the general algorithm, plus a web application.
## Web application
The web application is based on code from a previous Udacity project on Disaster message classification.  It's a Flask-based application that uses Jinja to generate HTML code via templates.   It has a very simple interface, that will allow you to select a file using a file chooser and once you submit your uploaded image, it will give you the dog breed prediction along with sample images to help identify races.  

In order to run locally, you just need to type `python run.py` in the app directory, and the app will load an http server, accesible on `http://localhost:3001`.   

There may be cases where because of overlapping features, the certainty of the prediction is lower or it even predicts a different breed, but you will be able to see why the algorithm may be confusing the breed.

## Directory Structure
