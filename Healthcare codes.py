#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Capstone Project - Healthcare
# Importing Libraries
import pandas as pd
import numpy as np


# In[2]:


h_data = pd.read_csv("health care diabetes.csv")


# In[3]:


# Project Task: Week 1


# In[4]:


# checking the shape of dataset
h_data.shape


# In[5]:


# Viewing the main heading of the dataset 
h_data.head()


# In[6]:


# Checking whether data contains any null values
h_data.isnull().any()


# In[7]:


# No null values in the dataset


# In[11]:


# Exploring the variables using histograms
# Importing required libraries 
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from matplotlib import style


# In[9]:


plt.hist(h_data['Pregnancies'])
plt.xlabel('Pregnancies')
plt.ylabel('Frequency')


# In[10]:


plt.hist(h_data['Glucose'])


# In[11]:


plt.hist(h_data['BloodPressure'])


# In[12]:


plt.hist(h_data['SkinThickness'])


# In[13]:


plt.hist(h_data['Insulin'])


# In[14]:


plt.hist(h_data['BMI'])


# In[15]:


plt.hist(h_data['DiabetesPedigreeFunction'])


# In[16]:


plt.hist(h_data['Age'])


# In[17]:


plt.hist(h_data['Outcome'])


# In[18]:


# Checking the variable for its integer or float values
h_data.info()


# In[19]:


h_data['Pregnancies'].value_counts().head(5)


# In[20]:


h_data['Glucose'].value_counts().head(5)


# In[21]:


h_data['BloodPressure'].value_counts().head(5)


# In[22]:


h_data['SkinThickness'].value_counts().head(5)


# In[23]:


h_data['Insulin'].value_counts().head(5)


# In[24]:


h_data['BMI'].value_counts().head(5)


# In[25]:


h_data['DiabetesPedigreeFunction'].value_counts().head(5)


# In[26]:


h_data['Age'].value_counts().head(5)


# In[7]:


# Target variable
h_data['Outcome'].value_counts()


# In[28]:


# Project Task: Week 2


# In[8]:


h_data.describe().transpose()


# In[9]:


# Checking the balance of the data by plotting the count of outcome by its value
 


# In[12]:


# Target variable
tc = h_data['Outcome'].value_counts()
tc


# In[13]:


plt.hist(h_data['Outcome'])
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Frequency of Target variable OUTCOME', fontdict= {'size' : 20, 'color' : 'green', 'weight': 'bold'})
plt.show()


# In[14]:


plt.pie(tc.values, labels = tc.index, autopct = '%.2f%%')
plt.ylabel('')
plt.title('Frequency of Target variable OUTCOME', fontdict= {'size' : 20, 'color' : 'green', 'weight': 'bold'})
plt.show()


# In[15]:


# From the frequency table of target variable and above charts we can say that data is slightly imbalance
# For balance data the ratio between positive and negative should be either 50/50 or 60/40 
# In the above case the ratio is 65/35 


# In[35]:


# Creating scatter charts between the pair of variables to understand the relationships


# In[16]:


# Importing required libraries
import seaborn as sns 


# In[37]:


# Glucose, BloodPressure, SkinThickness, Insulin, BMI


# In[38]:


col_with_0 = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']  


# In[39]:


# Exploring the variables using histogram
h_data.hist(column=col_with_0, rwidth=0.95, figsize=(15,8))


# In[40]:


sns.pairplot(h_data)


# In[41]:


# From the above scatter pair plot we can see that:
# There is strong positive correlation between Age and BloodPressure
# There is no correlation between Glucose and SkinThickness
# There is positive correlation between BMI and SkinThickness


# In[42]:


# Performing correlation analysis
# Univariate analysis


# In[43]:


# Importing required libraries
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


# In[44]:


x = h_data.iloc[:,0:8]
y = h_data.iloc[:,-1]


# In[45]:


# Applying SelectKBest class to extract top 6 best features
bestfeatures = SelectKBest(score_func=chi2, k=6)
fit = bestfeatures.fit(x,y)


# In[46]:


dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(x.columns)


# In[47]:


# Concat two DataFrames for better visualization
featureScores = pd.concat([dfcolumns, dfscores], axis=1)
featureScores.columns = ['col','Score']


# In[48]:


featureScores


# In[49]:


# Print the 6 best features
print(featureScores.nlargest(6,'Score'))


# In[50]:


# The correlation of each feature in the dataset
corrmat = h_data.corr()
top_corr_features = corrmat.index
corrmat


# In[51]:


# Creating heatmap to visually explore the correlation of each feature in dataset
plt.figure(figsize=(16,8))
sns.heatmap(h_data[top_corr_features].corr(), annot=True, cmap='GnBu')


# In[52]:


# We can drop this two features since they are not correlated with target variable(Outcome)
h_data.drop(['BloodPressure','DiabetesPedigreeFunction'], axis=1, inplace=True)


# In[53]:


h_data.head()


# In[54]:


# Project Task: Week 3


# In[55]:


# Strategies for model building-
# The Target variable(dependent variable) that is 'Outcome' is binary that is it has only two values 0 or 1
# As dependent variable(Target variable) is binary therefore Logistic regression is the appropriate regression analysis 


# In[56]:


# Importing required class
from sklearn.linear_model import LogisticRegression


# In[57]:


# Split the dataset in features and target varaible
x = h_data.drop("Outcome", axis=1)
y = h_data["Outcome"]


# In[58]:


# Importing required libraries to split x and y into training and testing set.
from sklearn.model_selection import train_test_split


# In[59]:


# We are splitting data in 75% of the data for model training and 25% for model testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)


# In[60]:


# Creating the model
logreg = LogisticRegression()

# fitting the model with train data
logreg.fit(x_train, y_train)

# fitting the model with predictor test data(x_test)
y_pred = logreg.predict(x_test)


# In[61]:


# Evaluating the model using confusion matrix


# In[62]:


# Importing the metrics for creating confusion matrix
from sklearn import metrics


# In[63]:


cf_matrix = metrics.confusion_matrix(y_test, y_pred)
cf_matrix


# In[64]:


# There were 117 True Positives, patients with diabetes that were correctly classified and 33 True Negatives, 
# patients without diabetes that were correctly classified

# The algorithm misclassified 29 patients that did have a diabetes by saying they did not (False Negative) and 
# algorithm misclassified 13 patients that did not have diabetes by saying that they did (False Positive)


# In[65]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))


# In[66]:


# Classification rate of 78% is considered as the good accuracy
# The model got 72% approx accuracy in predicting the patients with diabetes
# Recall percent of 53% indicates the results of the patients predicted to have diabetes and 
# Logistic regression can capture 53% of patients with diabetes


# In[67]:


# Comparing DecisionTree, Random Forest, and Logistic Regression classifier with KNN Algorithm


# In[68]:


# Importing required libraries (pipeline and models)
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


# In[69]:


# Creating pipeline for Decision Tree Classifier
pipeline_dt=Pipeline([('dt_classifier',DecisionTreeClassifier(random_state=0))])


# In[70]:


# Creating pipeline for Random Forest Classifier
pipeline_rf=Pipeline([('rf_classifier',RandomForestClassifier())])


# In[71]:


# Creating pipeline for KNeighbors Classifier
pipeline_knn=Pipeline([('kn_classifier',KNeighborsClassifier())])


# In[72]:


# Creating pipeline for Logistic Regression
pipeline_lr=Pipeline([('lr_classifier',LogisticRegression())])


# In[73]:


# Making the list of all pipelines
pipelines = [pipeline_dt,pipeline_rf,pipeline_knn,pipeline_lr]


# In[74]:


best_accuracy=0.0
best_classifier=0
best_pipeline=""


# In[75]:


# Dictionary of pipelines and classifier type for ease of reference
pipe_dict = {0: 'Decision Tree', 1: 'RandomForest', 2: 'KNeighbors', 3:'Logistic Regression'}

# Fitting the pipilines
for pipe in pipelines:
    pipe.fit(x_train, y_train)


# In[76]:


for i,model in enumerate(pipelines):
    print("{} Test Accuracy: {}".format(pipe_dict[i],model.score(x_test,y_test)))


# In[77]:


for i,model in enumerate(pipelines):
    if model.score(x_test,y_test)>best_accuracy:
        best_accuracy=model.score(x_test,y_test)
        best_pipeline=model
        best_classifier=i
print('Classifier with best accuracy:{}'.format(pipe_dict[best_classifier]))  


# In[78]:


# Logistic Regression is the best classifier compared with other algorithms, with accuracy of 78%, which is good


# In[79]:


# Creating Confusion Matrix for each Algorithm to compare each classifier


# In[80]:


# Decision Tree Confusion matrix
y_pred1 = pipeline_dt.predict(x_test)
dt_cnf_matrix=metrics.confusion_matrix(y_test,y_pred1)
dt_cnf_matrix


# In[81]:


# Random Forest Confusion matrix
y_pred2 = pipeline_rf.predict(x_test)
rf_cnf_matrix=metrics.confusion_matrix(y_test,y_pred2)
rf_cnf_matrix


# In[83]:


# KNeighbours Confusion matrix
y_pred3= pipeline_knn.predict(x_test)
knn_cnf_matrix=metrics.confusion_matrix(y_test,y_pred3)
knn_cnf_matrix


# In[84]:


# Logistic Regression Confusion matrix
y_pred4=pipeline_lr.predict(x_test)
lr_cnf_matrix=metrics.confusion_matrix(y_test,y_pred4)
lr_cnf_matrix


# In[85]:


# Decision Tree correctly classified 105 patients with diabetes and 41 patients without diabetes
# Random Forest correctly classified 115 patients with diabetes and 31 patients without diabetes
# KNN correctly classified 111 patients with diabetes and 37 patients without diabetes
# Logistic Regression correctly classified 117 patients with diabetes and 33 patients without diabetes


# In[86]:


# KNN and Logistic Regression are better performers as compared to Decision Tree and Random Forest


# In[87]:


# Project Task: Week 4


# In[88]:


# Creating a classification report by analyzing sensitivity, specificity, AUC (ROC curve)


# In[89]:


# Sensitivity identifies what percentage of patients with diabetes were correctlly identified.
# Specificity identifies what percentage of patients without diabetes were correctly identified.


# In[90]:


# As Logistic Regression and KNN are better performers therefore we are analyzing sensitivity, specificity, AUC (ROC curve) 
# on these two classifiers


# In[91]:


# For KNeighbors
sensitivity, specificity= [111/(111+25), 37/(37+19)]
print("sensitivity:", sensitivity)
print("specificity:", specificity)


# In[92]:


# For Logistic Regression
sensitivity, specificity= [117/(117+29), 33/(33+13)]
print("sensitivity:", sensitivity)
print("specificity:", specificity)


# In[94]:


# KNeighbours has sensitivity approx equivalent to Logistic Regresssion
# Logistic Regression has better specificity as compared to KNeighbours 
# Therefore i prefer Logistic regression over KNeighbours as it has both sensitivity and specificity better than other one


# In[95]:


# AUC (ROC curve) for Logistic Regression


# In[96]:


# Importing required libraries
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


# In[97]:


# We use roc_curve() to get the threshold, TPR and FPR
fpr, tpr, thresholds = roc_curve(y_test, pipeline_lr.predict_proba(x_test)[:,1])


# In[98]:


fpr


# In[99]:


tpr


# In[100]:


thresholds


# In[101]:


# For AUC we use roc_auc_score() function for ROC
lr_roc_auc = roc_auc_score(y_test, pipeline_lr.predict(x_test))


# In[102]:


#Plot the ROC Curve
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression(Sensitivity = %0.3f)' % lr_roc_auc)
plt.plot([0,1], [0,1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# In[ ]:




