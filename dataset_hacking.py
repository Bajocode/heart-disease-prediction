import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split as split
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
from sklearn import svm


""" wrangling
"""
data_url = ('https://archive.ics.uci.edu/ml/machine-learning-databases/'
            'heart-disease/processed.cleveland.data')
features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']
df = pd.read_csv(data_url, names=features)

print df.head()
print df.shape
print df.info()
df.replace('?', np.nan, inplace=True)
df.dropna(subset=['ca', 'thal'], axis=0, inplace=True)
df.reset_index(drop=True, inplace=True)
missing_data = df.isnull()

df['ca'] = pd.to_numeric(df['ca'], errors='coerce')
df[['age', 'sex', 'exang', 'ca']] = df[['age', 'sex', 'exang', 'ca']].astype(int)
df['num'].replace(to_replace=[1, 2, 3, 4], value=1, inplace=True)
df.rename(columns={'num': 'has_disease'}, inplace=True)

""" cexploritory data analysis
"""
print df.describe()
fig_has_disease, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))
sns.countplot(x='has_disease', data=df, ax=axes)

# categorial
# sex
fig_sex, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
sns.countplot(x='sex', data=df, ax=axes[0])
average_sex = df[["sex", "has_disease"]].groupby(['sex'], as_index=False).mean()
sns.barplot(x='sex', y='has_disease', data=average_sex, ax=axes[1])

# cp: chest pain type
fig_cp, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
sns.countplot(x='cp', data=df, ax=axes[0])
sns.countplot(x='has_disease', hue='cp', data=df, ax=axes[1])
average_cp = df[["cp", "has_disease"]].groupby(['cp'], as_index=False).mean()
sns.barplot(x='cp', y='has_disease', hue='cp', data=average_cp, ax=axes[2])

# fbs: fasting blood sugar > 120 mg (binary)
fig_fbs, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
sns.countplot(x='fbs', data=df, ax=axes[0])
sns.countplot(x='has_disease', hue='fbs', data=df, ax=axes[1])
average_fbs = df[["fbs", "has_disease"]].groupby(['fbs'], as_index=False).mean()
sns.barplot(x='fbs', y='has_disease', hue='fbs', data=average_fbs, ax=axes[2])

# restecg: resting electrocardiographic results
fig_restecg, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
sns.countplot(x='restecg', data=df, ax=axes[0])
sns.countplot(x='has_disease', hue='restecg', data=df, ax=axes[1])
average_restecg = df[["restecg", "has_disease"]].groupby(['restecg'], as_index=False).mean()
sns.barplot(x='restecg', y='has_disease', hue='restecg', data=average_restecg, ax=axes[2])

# slope: slope of peak excercie ST segment
fig_slope, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
sns.countplot(x='slope', data=df, ax=axes[0])
sns.countplot(x='has_disease', hue='slope', data=df, ax=axes[1])
average_slope = df[["slope", "has_disease"]].groupby(['slope'], as_index=False).mean()
sns.barplot(x='slope', y='has_disease', hue='slope', data=average_slope, ax=axes[2])

# thal: thalium stress test result
fig_thal, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
sns.countplot(x='thal', data=df, ax=axes[0])
sns.countplot(x='has_disease', hue='thal', data=df, ax=axes[1])
average_thal = df[["thal", "has_disease"]].groupby(['thal'], as_index=False).mean()
sns.barplot(x='thal', y='has_disease', hue='thal', data=average_thal, ax=axes[2])

# continuous
# age: peaks has disease
fig_age, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 8))
facet_grid = sns.FacetGrid(df, hue='has_disease')
facet_grid.map(sns.kdeplot, "age", shade=True, ax=axes[0])
legend_labels = ['Disease False', 'Disease True']
for t, l in zip(axes[0].get_legend().texts, legend_labels):
    t.set_text(l)
axes[0].set(xlabel='Age', ylabel='Density')

average_age = df[["age", "has_disease"]].groupby(['age'], as_index=False).mean()
sns.barplot(x='age', y='has_disease', data=average_age, ax=axes[1])
axes[1].set(xlabel='Age', ylabel='Average Disease Risk')

# restecg: resting electrocardiographic results
fig_trestbps, axes = plt.subplots(nrows=5, ncols=2, figsize=(8, 12))
sns.distplot(df[['trestbps']], ax=axes[0][0])
sns.violinplot(x='has_disease', y='trestbps', data=df, ax=axes[0][1])

# chol: serum cholestoral in mg/dl
sns.distplot(df[['chol']], ax=axes[1][0])
sns.violinplot(x='has_disease', y='chol', data=df, ax=axes[1][1])

# thalach: Max. heart rate achieved during thalium stress test
sns.distplot(df[['thalach']], ax=axes[2][0])
sns.violinplot(x='has_disease', y='thalach', data=df, ax=axes[2][1])

# oldpeak: ST depression induced by excercise relative to rest
sns.distplot(df[['oldpeak']], ax=axes[3][0])
sns.violinplot(x='has_disease', y='oldpeak', data=df, ax=axes[3][1])

# ca: Number of major vessels colored by fluoroscopy
sns.distplot(df[['ca']], ax=axes[4][0])
sns.violinplot(x='has_disease', y='ca', data=df, ax=axes[4][1])

# plt.show()

""" dummy variables
"""
cp_dummy = pd.get_dummies(df['cp'])
cp_dummy.rename(columns={1.0: 'cp_typical_angina', 2.0: 'cp_atypical_angina',
                         3.0: 'cp_non_angina', 4.0: 'cp_asymptomatic_angina'},
                inplace=True)
restecg_dummy = pd.get_dummies(df['restecg'])
restecg_dummy.rename(columns={0.0: 'restecg_normal', 1.0: 'restecg_wave_abnorm',
                              2.0: 'restecg_ventricular_ht'}, inplace=True)
slope_dummy = pd.get_dummies(df['slope'])
slope_dummy.rename(columns={1.0: 'slope_upsloping', 2.0: 'slope_flat',
                            3.0: 'slope_downsloping'}, inplace=True)
thal_dummy = pd.get_dummies(df['thal'])
thal_dummy.rename(columns={'3.0': 'thal_normal', '6.0': 'thal_fixed_defect',
                           '7.0': 'thal_reversible_defect'}, inplace=True)
df = pd.concat([df, cp_dummy, restecg_dummy, slope_dummy, thal_dummy], axis=1)
df.drop(['cp', 'restecg', 'slope', 'thal'], axis=1, inplace=True)

df_X = df.drop('has_disease', axis=1)
df_y = df['has_disease']
df_X_train, df_X_test, df_y_train, df_y_test = split(df_X, df_y, test_size=0.3, random_state=0)
kfold = model_selection.KFold(n_splits=10, random_state=7)
scoring = 'accuracy'

""" logistic regression
"""
lr = LogisticRegression()
rfe = RFE(lr)
rfe = rfe.fit(df_X.values, df_y.values)
features = []
for i, col in enumerate(df_X.columns.values):
    if rfe.support_[i]:
        features.append(col)
lr_X = df_X[features]
lr_y = df_y
# feature significance
lm = sm.Logit(lr_y, lr_X)
result = lm.fit()
print result.summary()
# modeling
X_train, X_test, y_train, y_test = split(lr_X, lr_y, test_size=0.3, random_state=0)
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
print 'Logistic regression accuracy: ', lr.score(X_test, y_test)
lr_cv = LogisticRegression()
kfold_result = model_selection.cross_val_score(lr_cv, X_train, y_train, cv=kfold, scoring=scoring)
print 'Logistic regression accuracy kfold: ', kfold_result.mean()

""" random forests
"""
features = df_X.columns.values
clf = RandomForestClassifier(n_estimators=20, random_state=0, n_jobs=1)
clf.fit(df_X_train, df_y_train)
print sorted(zip(features, clf.feature_importances_), key=lambda x: x[1], reverse=True)
sfm = SelectFromModel(clf, threshold=0.05)
sfm.fit(df_X_train, df_y_train)

for feature_index in sfm.get_support(indices=True):
    print features[feature_index]

X_train = sfm.transform(df_X_train)
X_test = sfm.transform(df_X_test)
clf_selected = RandomForestClassifier(n_estimators=20, random_state=0, n_jobs=1)
clf_selected.fit(X_train, df_y_train)
y_pred = clf_selected.predict(X_test)
print 'Random forests accuracy: ', accuracy_score(df_y_test, y_pred)

""" svm
"""
# from sklearn.model_selection import GridSearchCV
# parameters = [{'kernel': ['rbf'], 'gamma': [1e-4, 1e-3, 0.01, 0.1, 0.2, 0.5],
#               'C': [1, 10, 100, 1000]},
#               {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
# clf = GridSearchCV(svm.SVC(decision_function_shape='ovr'), parameters, cv=5)
# clf.fit(df_X_train, df_y_train)
# means = clf.cv_results_['mean_test_score']
# stds = clf.cv_results_['std_test_score']
# for mean, std, params in zip(means, stds, clf.cv_results_['params']):
#     print("%0.3f (+/-%0.03f) for %r"
#           % (mean, std * 2, params))
# print()
