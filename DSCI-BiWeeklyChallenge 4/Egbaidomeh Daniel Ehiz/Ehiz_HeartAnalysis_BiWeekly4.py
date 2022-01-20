#!/usr/bin/env python
# coding: utf-8

# In[47]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[48]:


ds = pd.read_csv(r"C:\Users\HP\Desktop\stutern_school\DSCI-BiWeeklyChallenge 4/heart_failure_clinical_records_dataset.csv")


# -------------------------------------------------------------
# #### _Feature Selection_

# In[49]:


pd.DataFrame(ds)


# #### _Turn code to a Data Cleaning Model_

# In[50]:


def prepossesing_data(data):
    data = data

    # Drop unnecessary column
    # data2 = data.drop('time', axis = 1, inplace = True)

    # Classify age into groups(40-55, 55-65, 65-75, 75-100)
    age_cut = pd.cut(data['age'], bins = [39,55,65,75,100], labels= [0, 1, 2, 3])
    data.insert(0, 'age_class', age_cut)
    data3 = data.drop(['age'], axis = 1, inplace= True)

    return data3


# In[51]:


prepossesing_data(ds)
ds.head()


# ---------------------------------------------------
# #### _Test and Train_

# In[52]:


import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import confusion_matrix


# In[53]:


# Split into train and test data
ds_train, ds_test = train_test_split(ds, train_size = 0.85, test_size = 0.15, random_state = 32) 


# In[54]:


ds_train.head()


# In[55]:


features = (['age_class', 'anaemia', 'creatinine_phosphokinase', 'diabetes',
       'ejection_fraction', 'high_blood_pressure', 'platelets',
       'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time','DEATH_EVENT'])


# In[56]:


# Standardize using MinMaxScaler
scaler = MinMaxScaler()
ds_train[features] = scaler.fit_transform(ds_train[features])


# In[57]:


# Instantiate the x and y train
Y_train = ds_train.pop('DEATH_EVENT')
X_train = ds_train


# In[58]:


# convert to n umpy array 
y_train = Y_train.to_numpy()
x_train = X_train.to_numpy()


# -------------------------------------------------------------

# #### _Logistic Regression_

# In[59]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve


# In[60]:


scaler = MinMaxScaler()
ds_test[features] = scaler.fit_transform(ds_test[features])


# In[61]:


Y_test = ds_test.pop('DEATH_EVENT')
X_test = ds_test


# In[62]:


# convert to numpy array
y_test = Y_test.to_numpy()
x_test = X_test.to_numpy()


# In[63]:


clf = LogisticRegression (solver = "liblinear")
parameters = {'penalty': ['l1', 'l2'], 'C' :[0.001, 0.009, 0.01, 0.09, 1, 5], 'max_iter': [150, 200, 250, 300, 350]}
grid_search =GridSearchCV(clf, param_grid = parameters, scoring = 'f1', verbose = 0, n_jobs = -1, cv = 5)
grid_search.fit(x_train, y_train)


# In[64]:


# optimal parameters
grid_search.best_params_


# In[65]:


model = LogisticRegression(solver = 'liblinear', max_iter = 150, penalty = 'l1')
model.fit(x_train, y_train)


# In[66]:


Regression_pred = model.predict(x_test)


# In[67]:


# Classification Report
from sklearn.metrics import classification_report
print (classification_report (y_test, Regression_pred ))


# In[68]:


# Save the model using joblib
import joblib

Logistic_model = "finalized_model.sav"
joblib.dump(model, Logistic_model)


# -----------------------------------------------------------------------------------------------------

# ##### _Support Vector Model (SVM)_

# In[69]:


from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
# model2 = SVC(kernel = 'linear')


# In[70]:


# HyperParameter tunning using GridSearchCV

parameters = [{'C':[1, 10, 100, 1000], 'kernel':['linear'] },
            {'C':[1, 10, 100, 1000], 'kernel':['rbf'], 'gamma' : [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
grid_search2 = GridSearchCV (estimator = SVC(),
             param_grid = parameters,
             scoring = 'accuracy',
             cv = 10,
             n_jobs = -1)

grid_search2 = grid_search2.fit(x_train, y_train)


# In[71]:


# Check the accuracy of the tunning

param_score = grid_search2.best_score_
param_score


# In[72]:


# optimal parameters
grid_search2.best_params_


# In[73]:


# Input the best parameters into the model

model2 = SVC(kernel = 'rbf', gamma = 0.9)
model2.fit(x_train, y_train)


# In[74]:


SVM_pred = model2.predict(x_test)


# In[75]:


mae = mean_absolute_error( y_test, SVM_pred)
print(mae)


# In[97]:


# Classification Result for Support Vector Model
print (classification_report (y_test, SVM_pred ))


# #### _TensorFlow Model_

# In[77]:


import tensorflow as tf


# In[78]:


tf.convert_to_tensor(ds_train)


# In[79]:


# We use the normalized layer as the first input layer of a simple model
def basic_model():
    tf_model = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation = tf.nn.relu), #1st input layer
            tf.keras.layers.Dense(128, activation = tf.nn.relu),  # 2nd input layer
            tf.keras.layers.Dense(1, activation = tf.nn.sigmoid)]) # output layer
    
    # Define Parameters for training the model
    tf_model.compile(optimizer='adam',
                        loss='binary_crossentropy',
                        metrics=['accuracy'])
    return tf_model


# In[80]:


model3 = basic_model()
model3.fit(x_train, y_train, epochs = 20, batch_size = 32 )


# In[81]:


# Estimations to check the loss and accuracy
loss, accuracy = model3.evaluate(x_test, y_test)
print('estimated loss:{}'. format(loss))
print('estimated accuracy: {}'. format(accuracy))


# In[82]:


prediction = model3.predict([x_test], verbose = 0)
# np.round(prediction)


# In[83]:


model3.save('Heart_Case_Predict.model')


# -----------------------------------------------------
# #### _Confusion Matrices_

# In[96]:


# Confusion Matrix for Logistic Regression 

LogReg = confusion_matrix(y_test, Regression_pred)
LogReg

# The Confusion matrix shows that the metric was calculated on 45 samples (29+4+4+8)
# True positive has the highest value (29) and a relatively low true negetive (8)
# This simply insinuates that most of the predictions were rightly predicted.


# In[98]:


# Confusion Matrix for Logistic Regression 

SVM_Reg = confusion_matrix(y_test, SVM_pred)
SVM_Reg

# The Confusion matrix shows that the metric was calculated on 45 samples (29+4+5+7)
# True positive has the highest value (29) and a relatively low true negetive (7)
# This simply insinuates that most of the predictions were rightly predicted.


# -------------------------------------------------------
# #### _Plot ROC and AUC_

# In[ ]:


# Confusion Matrix for the 


# In[101]:


from sklearn.metrics import roc_curve, auc

Logistic_fpr, Logistic_tpr, threshold = roc_curve(y_test, Regression_pred)
auc_Logistic = auc(Logistic_fpr, Logistic_tpr)

SVM_fpr, SVM_tpr, threshold = roc_curve(y_test, SVM_pred)
auc_SVM = auc(SVM_fpr, SVM_tpr)

plt.figure(figsize=(7,5))
plt.plot(SVM_fpr, SVM_tpr, linestyle = '-', label = 'SVM(auc = %0.3f)' % auc_SVM)
plt.plot(Logistic_fpr, Logistic_tpr, marker = '_', label = 'Logistic(auc = %0.3f)' % auc_Logistic)

plt.xlabel('false positive rate -->')
plt.ylabel('true positive rate -->')

plt.legend()

plt.show()


# In[ ]:


# To be able to rewad the new model
# new_model = tf.keras.models.load_model('Heart_Case_Predict.model')


# In[ ]:


# # To run this model on a new dataset
# predictions = new_model.predict(['new dataset'])
# # the new dataset are all x_test


# ----------------------------------------------
# ### _Feature Selection_ 

# In[84]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

best_feature = SelectKBest(score_func = chi2, k = 10) # instantiate the algo
fitting = best_feature.fit(X_train, Y_train) # fit the algo to dataset

dsScores = pd.DataFrame(fitting.scores_) # calculates the scores of every column
dsColumns = pd.DataFrame(X_train.columns) # stores the column names to dsColumns

featureScores = pd.concat([dsColumns, dsScores], axis = 1) # Concat 'dsScores' and 'dsColumns' for visibility
featureScores.columns = ['spec', 'score'] # names the columns in the concat 

# Print the top 5 features
new_feature = featureScores.nlargest(5, 'score')
print(new_feature)


# In[85]:


# Standardize the new feature
new_feature_score = (new_feature['score'] - new_feature['score'].mean())/ new_feature['score'].std()


# In[86]:


new_feature_score


# In[87]:


# 'time', 'age_class', 'serum_creatinine', 'ejection_fraction', 'high_blood_pressure' are the Top 5 best features
new_feature_numpy = np.asarray(new_feature_score) # convert to numpy


# In[88]:


# The data was trained on a 300 row and 13 column
# for the model to read this new data having 5 rows, it needs to be reshaped

reshaped_new_feature= new_feature_numpy.reshape(1, -1) 


# In[ ]:


LoadedLogisticPrediction = joblib.load(Logistic_model)


# In[ ]:


# result = LoadedLogisticPrediction.score(reshaped_new_feature, y_test) 


# In[ ]:




