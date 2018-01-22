# Machine Learning Engineer Nanodegree

## Capstone Proposal - Classification
Jordan Carson  
January 15th, 2018


### Domain Background
According to Kaggle, “Every minute, the world loses an area of forest the size of 48 football fields. And deforestation in the Amazon Basin accounts for the largest share, contributing to reduced biodiversity, habitat loss, climate change, and other devastating effects. But better data about the location of deforestation and human encroachment on forests can help governments and local stakeholders respond more quickly and effectively.”

In this project, I will be creating a classification system to label satellite image chips with atmospheric conditions and various classes of land cover/land use. This is related to the Kaggle competition titled ‘Planet: Understanding the Amazon from Space’.

### Problem Statement

Given a dataset of satellite images for the Amazon, an algorithm needs to be developed to classify each image listed in the test set with a tag that is believed to be associated to the image. There are 17 possible tags: agriculture, artisinal_mine, bare_ground, blooming, blow_down, clear, cloudy, conventional_mine, cultivation, habitation, haze, partly_cloudy, primary, road, selective_logging, slash_burn, water. 

The goal of this project is to train and build a deep learning model that is able to recognize an entire dataset of imagery from the amazon and properly label it with the appropriate tagging. 

### Datasets and Inputs

The datasets for this project are all located in Kaggle. The data-set comes in a train & test set split already defined for use. 

According to Kaggle, “The chips for this competition were derived from Planet's full-frame analytic scene products using our 4-band satellites in sun-synchronous orbit (SSO) and International Space Station (ISS) orbit. The set of chips for this competition use the GeoTiff format and each contain four bands of data: red, green, blue, and near infrared. The specific spectral response of the satellites can be found in the Planet documentation. Each of these channels is in 16-bit digital number format, and meets the specification of the Planet four band analytic ortho scene product."


### Solution Statement

I plan to build and implement a deep learning model for the final solution. This is because deep learning models should be more effective at determining the relative features from a given image in our dataset rather than other computer vision techniques. 

A convolution neural network will be built that can be trained on the training data (imagery.) The deep learning network will be built using tensorflow and Keras.

### Benchmark Model

The benchmark deep learning model will be a convolution neural network that will be used to obtain a high accuracy rating similar to the individuals who classified in the top 20 tier of the Kaggle competition (on the public leaderboard). This is an ambitious goal.

### Evaluation Metrics


### Project Design

