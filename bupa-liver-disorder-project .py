#!/usr/bin/env python
# coding: utf-8

# In[4]:


# for numerical computing
import numpy as np

# for dataframes
import pandas as pd

# for easier visualization
import seaborn as sns

# for visualization and to display plots
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# import color maps
from matplotlib.colors import ListedColormap

# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")

from math import sqrt

# To split train and test sets
from sklearn.model_selection import train_test_split

# To perform hyperparameter tuning
from sklearn.model_selection import GridSearchCV, cross_val_score

# Machine Learning Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Metrics
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix, accuracy_score

# Preprocessing
from sklearn.preprocessing import StandardScaler

# Plotting
from xgboost import plot_importance  # To plot feature importance
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# To save the final model on disk
import joblib


# In[5]:


# read the dataset
df = pd.read_csv('liver.csv')


# In[6]:


df.head()

#Target column is Dataset
#Supervised
#Classification
#Binary classification
# In[6]:


df.columns


# In[7]:


df.shape


# # Exploratory Data Analysis

# In[10]:


df.describe()


# It seems there is outlier in Aspartate_Aminotransferase as the max value is very high than mean value

# In[11]:


df.dtypes


# In[12]:


df.info()


# #Filtering categorical data

# In[13]:


df.dtypes[df.dtypes=='object']


# # Distribution of numerical Features

# In[14]:


df.hist(figsize=(15,15),xrot = -45, bins=10)
plt.show()


# Dataset i.e output value has '1' for liver disease and '2' for no liver disease so let's make it 0 for no disease to make it convenient

# In[15]:


#converting values in a Dataset column
def convertdataset(x):
    if x == 2:
        return 0
    return 1

# Apply function to 'Dataset' column
df['Dataset'] = df['Dataset'].map(convertdataset)


# In[16]:


df.head()


# In[18]:


df.Dataset.value_counts()
## '1' --> liver disease
##'0' --> no liver disease


# # Distribution of categorical data

# In[20]:


df.describe(include=['object'])


# In[21]:


#Bar plots for categorical features
plt.figure(figsize=(5,5))
sns.countplot(y='Gender',data = df)


# In[22]:


df[df['Gender'] =='Male'][['Dataset','Gender']].head()


# In[23]:


sns.catplot(x="Age", y="Gender", hue="Dataset", data=df, kind="strip")
plt.show()


# In[24]:


df['Gender'].value_counts()


# In[25]:


sns.countplot(data=df, x = 'Gender', label='Count')

M, F = df['Gender'].value_counts()
print('Number of patients that are male: ',M)
print('Number of patients that are female: ',F)

Label Male as 0 and Female as 1
# In[26]:


#Categorical value handling
def convertgender(x):
    if x=='Male':
        return 0
    return 1
df['Gender'] = df['Gender'].map(convertgender)


# In[27]:


df.head()


# # Correlations
#Positive Correlation --> one feature increases other also increases
#Negative Correlation --> one feature increases other decreases
# closer to 0 --> weak relationship
# In[29]:


#Correlation Analysis
df.corr()


# In[30]:


plt.figure(figsize = (10,10))
sns.heatmap(df.corr())


# In[31]:


# Creating a mask for the upper triangle
mask = np.zeros_like(df.corr())
mask[np.triu_indices_from(mask)] = True

# Setting up the figure
plt.figure(figsize=(10,10))

# Plotting the heatmap with a specific style
with sns.axes_style("white"):
    ax = sns.heatmap(df.corr()*100, mask=mask, fmt=".0f", annot=True, lw=1, cmap=ListedColormap(['green','yellow','red','blue']))

# Show the plot
plt.show()


# # Data Cleaning

# In[33]:


df= df.drop_duplicates()
print(df.shape)

There were 13 duplicates
# In[34]:


df.columns


# ## Removing Outlier

# In[35]:


sns.boxplot(df.Aspartate_Aminotransferase)


# In[37]:


df.Aspartate_Aminotransferase.sort_values(ascending=False).head()


# In[38]:


df = df[df.Aspartate_Aminotransferase<=3000]


# In[39]:


df.shape


# In[40]:


df.Aspartate_Aminotransferase.sort_values(ascending=False).head()


# In[41]:


df = df[df.Aspartate_Aminotransferase<=2500]


# In[42]:


sns.boxplot(df.Aspartate_Aminotransferase)


# In[43]:


df.shape


# In[45]:


df.isnull().sum()


# In[46]:


df = df.dropna(how='any')


# In[47]:


df.head()


# In[48]:


df.shape


# # Machine Learning Models
# Data preparation 
# In[49]:


# Create separate object for target variable
y = df.Dataset

# Create separate object for input features
X=df.drop('Dataset',axis=1)


# In[50]:


#Split X and y into train and test sets 
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state= 42, stratify=y)


# In[51]:


# Print number of observations in X_train, X_test, y_train, and y_test
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# # Normalization

# In[52]:


# Normalization using StandardScaler
scaler = StandardScaler()

# Fit the scaler on the training data and transform both training and test data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Now X_train_scaled and X_test_scaled are normalized versions of your data


# In[53]:


X_train_scaled


# In[56]:


X_test_scaled


# # Model-Selection 

# In[68]:


from sklearn.model_selection import cross_val_score

# Perform cross-validation
model_score = cross_val_score(estimator=LogisticRegression(), X=X_train_scaled, y=y_train, cv=5)

# Print the cross-validation scores for each fold
print(model_score)

# Print the mean cross-validation score
print(model_score.mean())


# In[71]:


from sklearn.model_selection import cross_val_score
model_score =cross_val_score(estimator=RandomForestClassifier(),X=X_train_scaled, y=y_train, cv=5)
print(model_score)
print(model_score.mean())


# In[72]:


from sklearn.model_selection import cross_val_score
model_score =cross_val_score(estimator=XGBClassifier(),X=X_train_scaled, y=y_train, cv=5)
print(model_score)
print(model_score.mean())


# In[75]:


# Create param dictionary for LogisticRegression
model_param = {
    'LogisticRegression': {
        'model': LogisticRegression(max_iter=1000),  # Ensure a sufficient number of iterations
        'param': {
            'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],  # Regularization strength
            'penalty': ['l1', 'l2'],  # Type of regularization
            'solver': ['liblinear', 'lbfgs', 'newton-cg', 'saga']  # Optimization algorithms
        }
    },
    'RandomForestClassifier': {
        'model': RandomForestClassifier(),
        'param': {
            'n_estimators': [10, 50, 100, 130],
            'criterion': ['gini', 'entropy'],
            'max_depth': range(2, 4, 1),
            'max_features': ['auto', 'log2']
        }
    },
    'XGBClassifier': {
        'model': XGBClassifier(objective='binary:logistic'),
        'param': {
            'learning_rate': [0.5, 0.1, 0.01, 0.001],
            'max_depth': [3, 5, 10, 20],
            'n_estimators': [10, 50, 100, 200]
        }
    }
}


# In[76]:


# List to store results
scores = []

# Perform Grid Search for each model
for model_name, mp in model_param.items():
    model_selection = GridSearchCV(
        estimator=mp['model'], 
        param_grid=mp['param'], 
        cv=5, 
        return_train_score=False
    )
    model_selection.fit(X, y)
    scores.append({
        'model': model_name,
        'best_score': model_selection.best_score_,
        'best_params': model_selection.best_params_
    })

# Print the results
for score in scores:
    print(f"Model: {score['model']}")
    print(f"Best Score: {score['best_score']}")
    print(f"Best Params: {score['best_params']}")
    print()


# # Model Building

# In[78]:


#as per above results, logistic Regression gives best result and hence selecting same to model building...

# Define the Logistic Regression model with the best parameters
model_lr = LogisticRegression(C=0.1, penalty='l2', solver='lbfgs', max_iter=1000)

# Fit the model on your training data
model_lr.fit(X_train_scaled, y_train)

# Optionally, evaluate the model on test data
from sklearn.metrics import accuracy_score, classification_report

# Predict on the test set
y_pred = model_lr.predict(X_test_scaled)

# Print performance metrics
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print("Classification Report:")
print(classification_report(y_test, y_pred))


# In[80]:


# Get the feature importances (coefficients) from the Logistic Regression model
feature_importances = model_lr.coef_[0]  # For binary classification, coef_ is a 1D array
headers = ["name", "score"]
values = sorted(zip(X_train.columns, feature_importances), key=lambda x: x[1] * -1)

# Create a DataFrame for feature importances
lr_feature_importances = pd.DataFrame(values, columns=headers)

# Plot feature importances
fig = plt.figure(figsize=(15, 7))
x_pos = np.arange(len(lr_feature_importances))
plt.bar(x_pos, lr_feature_importances['score'])
plt.xticks(x_pos, lr_feature_importances['name'], rotation=90)
plt.title('Feature Importances (Logistic Regression)')
plt.xlabel('Features')
plt.ylabel('Coefficient Value')
plt.show()


# In[81]:


#Confusion Matrix
cm = confusion_matrix(y_test,model_lr.predict(X_test_scaled))
cm


# In[82]:


#plot the graph
from matplotlib import pyplot as plt
import seaborn as sn
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('True Value')
plt.show()

