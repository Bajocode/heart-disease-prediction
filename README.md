# Heart Disease Risk Estimation

## Introduction

### Data

The goal of this project is generating a model to
estimate the risk of having a heart disease. [The UCI heart disease
database](https://archive.ics.uci.edu/ml/machine-learning-databases/heart-
disease/) contains 76 attributes, but all published experiments refer to using a
subset of 14 of them. The target variable is an integer valued from 0 (no
presence) to 4. However, for sake of simplicity, this will be reduced to a
binary prediction, i.e, heart disease vs no heart disease. The dataset was
collected from the four following locations:

1. Cleveland Clinic Foundation
(cleveland.data)
2. Hungarian Institute of Cardiology, Budapest (hungarian.data)
3. V.A. Medical Center, Long Beach, CA (long-beach-va.data)
4. University
Hospital, Zurich, Switzerland (switzerland.data)

Each database has the same
instance format.  While the databases have 76 raw attributes, only 14 of them
are actually used.

*The authors of the databases: Hungarian Institute of
Cardiology. Budapest: Andras Janosi, M.D. University Hospital, Zurich,
Switzerland: William Steinbrunn, M.D. University Hospital, Basel, Switzerland:
Matthias Pfisterer, M.D. V.A. Medical Center, Long Beach and Cleveland Clinic
Foundation: Robert Detrano, M.D., Ph.D.*

### Flow

[Data fetching](#data-
fetching) --> [Wrangling](#wrangling) --> [Data analysis]() --> [Modeling]() -->
[evaluation]()

## Python imports

```python
%matplotlib inline

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.feature_selection import RFE
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split as split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn import svm 

sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
```

## Data fetching

```python
link_cleveland = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'
link_hungarian = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data'
link_swiss = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.switzerland.data'
link_veniceb = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.va.data'

links = [link_cleveland, link_hungarian, link_swiss, link_veniceb]
names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang',
 'oldpeak', 'slope', 'ca', 'thal', 'target']

df = pd.concat(map(lambda x: pd.read_csv(x, names=names), links))

df.head()
```

## Wrangling

### Handling missing values

```python
df.replace('?', np.nan, inplace=True)
df.isnull().sum()
```

```python
df.dropna(axis=0, inplace=True)
df.reset_index(drop = True, inplace = True)
```

### Correcting data formats

```python
df['ca'] = pd.to_numeric(df['ca'], errors='coerce')
df['thal'] = pd.to_numeric(df['thal'], errors='coerce')
df[['age', 'sex', 'cp', 'fbs', 'restecg', 'exang', 'ca', 'slope', 'thal']] = df[['age', 'sex', 'cp', 'fbs', 'restecg', 'exang', 'ca', 'slope', 'thal']].astype(int)
df[['trestbps', 'chol', 'thalach', 'oldpeak']] = df[['trestbps', 'chol', 'thalach', 'oldpeak']].astype(float)
df['target'].replace(to_replace=[1, 2, 3, 4], value=1, inplace=True)
```

## Exploritory data analysis

### Target

```python
fig_target, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))
sns.countplot(x='target', data=df, ax=axes)
```

### Categorial

```python
def plotCategorial(attribute):
    fig_cp, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 4))
    sns.countplot(x=attribute, data=df, ax=axes[0])
    sns.countplot(x='target', hue=attribute, data=df, ax=axes[1])
    avg = df[[attribute, 'target']].groupby([attribute], as_index=False).mean()
    sns.barplot(x=attribute, y='target', hue=attribute, data=avg, ax=axes[2])
```

```python
plotCategorial('sex')
```

```python
plotCategorial('cp')
```

```python
plotCategorial('fbs')
```

```python
plotCategorial('restecg')
```

```python
plotCategorial('exang')
```

```python
plotCategorial('slope')
```

```python
plotCategorial('thal')
```

### Continuous

```python
def plotContinuous(attribute, ax_index):
    sns.distplot(df[[attribute]], ax=axes[ax_index][0])                                   
    sns.violinplot(x='target', y=attribute, data=df, ax=axes[ax_index][1])
```

```python
continuous = ['trestbps', 'chol', 'thalach', 'oldpeak', 'ca']
fig_continuous, axes = plt.subplots(nrows=len(continuous), ncols=2, figsize=(15, 22))
[plotContinuous(x, i) for i, x in enumerate(continuous)] 
```

#### Age

```python
fig_age, axes = plt.subplots(nrows=2, ncols=1, figsize=(15, 8))
facet_grid = sns.FacetGrid(df, hue='target')
facet_grid.map(sns.kdeplot, "age", shade=True, ax=axes[0])
avg = df[["age", "target"]].groupby(['age'], as_index=False).mean()
sns.barplot(x='age', y='target', data=avg, ax=axes[1])
plt.clf()
```

## Finalize data for modeling

### Dummy variables

```python
cp_dummy = pd.get_dummies(df['cp'])
cp_dummy.rename(columns={1: 'cp_typical_angina', 2: 'cp_atypical_angina',
                         3: 'cp_non_angina', 4: 'cp_asymptomatic_angina'},
                inplace=True)
restecg_dummy = pd.get_dummies(df['restecg'])
restecg_dummy.rename(columns={0: 'restecg_normal', 1: 'restecg_wave_abnorm',
                              2: 'restecg_ventricular_ht'}, inplace=True)
slope_dummy = pd.get_dummies(df['slope'])
slope_dummy.rename(columns={1: 'slope_upsloping', 2: 'slope_flat',
                            3: 'slope_downsloping'}, inplace=True)
thal_dummy = pd.get_dummies(df['thal'])
thal_dummy.rename(columns={3: 'thal_normal', 6: 'thal_fixed_defect',
                           7: 'thal_reversible_defect'}, inplace=True)
df = pd.concat([df, cp_dummy, restecg_dummy, slope_dummy, thal_dummy], axis=1)
df.drop(['cp', 'restecg', 'slope', 'thal'], axis=1, inplace=True)
```

### Separate target

```python
df_X = df.drop('target', axis=1)
df_y = df['target']
```

### Feature selection

```python
rfe = RFE(LogisticRegression())
rfe.fit(df_X.values, df_y.values)
selected_features = []

for i, col in enumerate(df_X.columns.values):
    if rfe.support_[i]:
        selected_features.append(col)

selected_X = df_X[selected_features]
selected_y = df_y

lm = sm.Logit(selected_y, selected_X)
result = lm.fit()

print result.summary()
```

```python
selected_X_train, selected_X_test, selected_y_train, selected_y_test = split(selected_X, selected_y, test_size=0.3, random_state=0)
```

## Modeling

### Logistic regression

```python
lr = LogisticRegression()
lr.fit(selected_X_train, selected_y_train)

print 'Accuracy: %.3f' % lr.score(selected_X_test, selected_y_test)
```

### Support Vector Machine

#### Tuning

```python
parameters = [{'kernel': ['rbf'], 'gamma': [1e-4, 1e-3, 0.01, 0.1, 0.2, 0.5],
               'C': [1, 10, 100]},
              {'kernel': ['linear'], 
               'C': [1, 10, 100]}]
grid = GridSearchCV(svm.SVC(decision_function_shape='ovr'), parameters, cv=5)
grid.fit(selected_X_train, selected_y_train)
grid_means = grid.cv_results_['mean_test_score']
grid_stds = grid.cv_results_['std_test_score']

for mean, std, params in zip(grid_means, grid_stds, grid.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))
```

```python
svm_linear = svm.SVC(kernel='linear', C=10)
svm_linear.fit(selected_X_train, selected_y_train)

print 'Accuracy: %.3f' % svm_linear.score(selected_X_test, selected_y_test)
```

### Cross validation

```python
kfold = model_selection.KFold(n_splits=10, random_state=7)
models = [('Linear regression', lr), 
          ('Support vector machine', svm_linear)]

for model in models:
    results = model_selection.cross_val_score(model[1], 
                                              selected_X_train, 
                                              selected_y_train, 
                                              cv=kfold, 
                                              scoring='accuracy')
    print 'Cross validated', model[0], 'Accuracy: %.3f' % results.mean()
```
