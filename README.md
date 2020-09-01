# **Bike sharing system**

Data source : https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset 

We are performed analysis on this dataset which contains the hourly count of rental bikes between years 2011 and 2012 in Capital bikeshare system with the corresponding weather and seasonal information. Dataset consists of 17379 rows and 17 features (columns).

## **Abstract**

Bike sharing systems are new generation of traditional bike rentals where whole process from membership, rental and return back has become automatic. Through these systems, user is able to easily rent a bike from a particular position and return back at another position. Currently, there are about over 500 bike-sharing programs around the world which is composed of over 500 thousands bicycles. Today, there exists great interest in these systems due to their important role in traffic, environmental and health issues.

This readme describes how the analysis is perfomed on this bike sharing systems dataset.

## **Objective** 


In our analysis, we will perform descriptive analysis and visualized the dataset to find vauable insights and build  regression model to predict count of the bikes rented.

It will help the emplyoee, residentsn students and tourists .


## **Methodology**

**Step 1** - Exploratory Data Analysis (EDA)

**Step 2** - Data Preprocessing

**Step 3** - Fitting Models

**Step 4** - Performance Summary & Model Selection


## Tools used

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

## Features (columns) of the dataset:

- instant : record index
- dteday : date
- season : season (1:winter, 2:spring, 3:summer, 4:fall)
- yr : year (0: 2011, 1:2012)
- mnth : month ( 1 to 12)
- hr : hour (0 to 23)
- holiday : weather day is holiday or not
- weekday : day of the week
- workingday : if day is neither weekend nor holiday is 1, otherwise is 0.
- weathersit :
    - 1 : Clear, Few clouds, Partly cloudy, Partly cloudy
    - 2 : Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
    - 3 : Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
    - 4 : Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
- temp : Normalized temperature in Celsius. The values are derived via (t-t_min)/(t_max-t_min), t_min=-8, t_max=+39 (only in hourly scale)
- atemp : Normalized feeling temperature in Celsius. The values are derived via (t-t_min)/(t_max-t_min), t_min=-16, t_max=+50 (only in hourly scale)
- hum : Normalized humidity. The values are divided to 100 (max)
- windspeed : Normalized wind speed. The values are divided to 67 (max)
- casual : count of casual users
- registered : count of registered users
- cnt : count of total rental bikes including both casual and registered


## Data Preprocessing

Data is often taken from sources which are normally not too reliable and that too in different formats, more than half our time is consumed in dealing with data quality issues when working on a machine learning problem. It is simply unrealistic to expect that the data will be perfect. There may be problems due to human error, limitations of measuring devices, or flaws in the data collection process.

* Misssing Values
It is very much usual to have missing values in your dataset. We've made sure there aren't any missing values.

* Inconsistent values
We know that data can contain inconsistent values. We have already faced this issue at some point. For instance, the ‘instant’ field is dropped since it represents the id's of the observation. 

* Multicollinearity 
It generally occurs when there are high correlations between two or more predictor variables. The fields _'temp'_ and _'atemp'_, _'mnth'_ and _'season'_ exhibit multicollinearity (redundancy) in the data. _'atemp'_ and _'season'_ are dropped as it can result in unstable and unreliable regression estimates.

## Fitting Models

Estimated distributed properties of the variables, potentially conditional on other variables. Summarized relationships between variabels and made inferential statements about those relationships.

### Linear Regression

Linear regression is perhaps one of the most well known and well understood algorithms in statistics and machine learning.
The representation is a linear equation that combines a specific set of input values (x) the solution to which is the predicted output for that set of input values (y). As such, both the input values (x) and the output value are numeric.
Root Mean Square Error (RMSE) is the standard deviation of the residuals (prediction errors). Residuals are a measure of how far from the regression line data points are; RMSE is a measure of how spread out these residuals are. In other words, it tells you how concentrated the data is around the line of best fit.
**Root Mean Square Error (RMSE) on test data: 151.6494699751178**

### K-nearest neighbors

K nearest neighbors is a simple algorithm that stores all available cases and classifies new cases based on a similarity measure (e.g., distance functions).
Performing 5-fold cross-validation on data and tuning hyperparameter 'n_neighbors' using grid search.
**KNN-119.84534328308183**

### Decision Tree

A decision tree is a flowchart-like structure in which each internal node represents a "test" on an attribute (e.g. whether a coin flip comes up heads or tails), each branch represents the outcome of the test, and each leaf node represents a class label (decision taken after computing all attributes).
Performing 5-fold cross-validation on data and tuning hyperparameter 'max_depth' using grid search.
**Root Mean Square Error (RMSE) on test data: 82.15474556915505**

### Random Forest

Random forests or random decision forests are an ensemble learning method for classification, regression and other tasks that operate by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees.
**Root Mean Square Error (RMSE) on test data: 66.31229301835563**

## Performance Summary & Model Selection

Model which gives lowest Root Mean Square Error (RMSE) value is preferred. From the above graph, Random Forest is giving better results on both train and test data. We will be using this random forest model (model_rf) for prediction of count of total rental bikes.**Root Mean Square Error (RMSE) on test data: 66.31229301835563**

## Aplication and Usage

* It will also enable to find mobility patterns in the city.
* Identifying the most of important events in the city by monitoring these data.
* This project could be expanded to Philadelphia's bike rental data analysis, since Philadelphia is well-known as bike-friendly city and many students living in University City use bike as transportation.
* Research purposes to discover important trends and relationships.

## Target Audience

* The possible target audience would be city employees who control over the bike rental business.
* Bike rental companies would be another possible target audience, because, in order to run the business, it is better to have statistical data which can explain and predicts the rentals including why people rent or when they rent more.
* Government can use this data to find mobility patterns in the city which can be helpul in decision-making process.
* Bike manufacturers may find this analysis useful.

## Limitation

* The dataset is from 2011 and 2012, so it is a bit old and only 2 years of data.
* Various other features such as the duration of travel, departure and arrival position, distance, etc. which can be helpful to fetch valuable insights are not present in the dataset.
* The dataset is not realtime one.

## Improvements

To improve, we can collect more years of data from various sources along with additional features.
