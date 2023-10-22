# Guvi_Final_Project

The problem statement has 4 projects. Refer attached PDF.
Dataset: https://drive.google.com/drive/folders/1_xO2V87HIcOeSw_0E4FR84Th1j1E4qxO?usp=share_link

## Twitter Sentiment Analysis

Dataset file name: twitter_new.csv

In this project, we are given with a dataset to train a model that will predict the sentiment of a comment (Positive or negative). The given dataset is already labeled.
We first do some pre-processing using various NLP techniques such as stemming and stopword removal. Then we use embedding techniques such as word2vec to convert this to numerical features.
We train the dataset on a decision tree model with cross-validation to obtain best accuracy of 0.9898505409292727.
Best hyperparameters for Decision tree: {'max_depth': 40, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 2}

## Course Rating Prediction

Dataset file name: 3.1-data-sheet-guvi-courses.csv

In this project we are given with a dataset to train a model that will predict the course rating. It is a regression problem with number of features.
We do some pre-processing and train the data on different models by using log, boxcox transformations.
Following are the results for various models-

1) Gradient boost Regressor on Random Forest Regressor using gridsearch cv- Best r2 score: 0.3393162654267248 and Best hyperparameters: {'learning_rate': 0.01, 'max_depth': 10, 'n_estimators': 200}
2) Adaboost Regressor on Random Forest Regressor using gridsearch cv- Best r2 score: 0.3706567575431674 and Best hyperparameters: {'base_estimator': RandomForestRegressor(max_depth=20, min_samples_split=10), 'learning_rate': 0.001, 'n_estimators': 100}
3) Degree 2 Polynomial regressor- r2 score: 0.17106278641848915

The best model obtained is with Adaboost Regressor with a pretty low r2 score. The given dataset seems to be improper

## Instagram Influencers

Dataset file name: Influencer.csv

Here we are given with simple EDA task and a number of questions are asked in the problem statement itself.
EDA is performed and all the answers are provided in the .ipynb file itself-

## Image Classification

Dataset link: https://www.kaggle.com/datasets/chetankv/dogs-cats-images?resource=download

We are to train a Dog vs Cat classification model using the dataset provided on kaggle.
We train 6 models and note their accuracies-

1) CNN: 85.05%
2) CNN with dropout=0.2: 86.2%
3) CNN with dropout=0.4: 85.4%
4) CNN with skip connection: 79.9%
5) CNN with skip connection and dropout=0.2: 80.35%
6) CNN with skip connection and dropout=0.4: 80.95%

Best results are obtained in CNN with dropout=0.2
