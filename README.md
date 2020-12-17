# Udacity Data Scientist Capstone Project
## Dog Breed Detection
> Francisco Chavez Clemente

# Project Definition
## Project Overview
One of the most popular pets in the world are dogs.  There are a large number of breeds and for all likings, and for someone who is not familiar with all breeds it may be difficult to identify the correct one for any given dog.  At the same time

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
In order to look at different pre-trained networks, I ran 100 epochs for each of the ones listed on the table, and the best two were the ResNet50 and the Xception networks, so I decided to use the Xception network as the pretrained network to get the highest possible accuracy.

| Model | Test accuracy |
| :-- | :--: |
| ResNet50 | 78.3493% |
| DogVGG16Data | 49.7608% |
| DogVGG19Data | 49.1627% |
| DogInceptionV3Data | 79.0670% | 
| DogXceptionData | 84.6890% |

# Deliverables
## Notable exclusions
In order to minimize the 
## Application
## Web application


## Directory Structure
