# Heart Disease Prediction

[📋 Complete Jupyter Notebook](./heart_disease_prediction.ipynb)

![](./header.jpg)

## Data

This project aims to generate a model to predict the presence of a heart disease. [The UCI heart disease database](https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/) contains 76 attributes, but all published experiments refer to using a subset of 14. The target attribute is an integer valued from 0 (no presence) to 4. However, for sake of simplicity it will be reduced to binary classification, i.e, `0` vs `0 <`. 

*The authors of the databases: Hungarian Institute of Cardiology. Budapest: Andras Janosi, M.D. University Hospital, Zurich, Switzerland: William Steinbrunn, M.D. University Hospital, Basel, Switzerland: Matthias Pfisterer, M.D. V.A. Medical Center, Long Beach and Cleveland Clinic Foundation: Robert Detrano, M.D., Ph.D.*

## Attributes

|  | Description | Variable | Type |
|:---------|:---------------------------------------------------------------------------------------------|:-----------|:--------|
| age | age in years | continuous | `int` |
| sex | 1 = male, 0 = female | categorial | `int` |
| cp | chest pain type: 1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic | categorial | `int` |
| trestbps | resting blood pressure in mm Hg | continuous | `float` |
| chol | serum cholestoral in mg/dl | continuous | `float` |
| fbs | fasting blood sugar > 120 mg/dl: 1 = true, 0 = false | categorial | `int` |
| restecg | 0: normal, 1: having ST-T wave abnormality, 2: left ventricular hypertrophy | categorial | `int` |
| thalach | maximum heart rate achieved | continuous | `float` |
| exang | exercise induced angina (1 = yes; 0 = no) | categorial | `int` |
| oldpeak | ST depression induced by exercise relative to rest | continuous | `float` |
| slope | the slope of the peak exercise ST segment: 1: upsloping, 2: flat, 3: downsloping | categorial | `int` |
| ca | number of major vessels: (0-3) colored by flourosopy | continuous | `int` |
| thal | 3: normal, 6: fixed defect, 7: reversable defect | categorial | `int` |
| target | diagnosis of heart disease: (0 = false, 1 = true | categorial | `int` |


## Flow

[Data fetching](#data-fetching) --> [Wrangling](#wrangling) --> [Data analysis]() --> [Modeling]() --> [evaluation]()
