#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

features_list = ['poi','deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees', 'to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi',  'shared_receipt_with_poi'] # You will need to use more features

#'from_poi_to_this_person',
#'from_this_person_to_poi',
#'shared_receipt_with_poi',
 # You will need to use more features

#features_list = ['poi','salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees', 'to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi',  'shared_receipt_with_poi'] # You will need to use more features
### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

### Task 2: Remove outliers
###MMP Note: regression
data = featureFormat(data_dict, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)





from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit(features,labels)
predictions = reg.predict(features)

print len(features)


def outlierCleaner(predictions, features, labels):

    cleaned_data = []

    ### your code goes here
    error=predictions-labels
    
    x=len(features)-1
    originallen=x
    while x>-1:

            cleaned_data.append((features[x],labels[x],error[x]))

            x=x-1

    cleaned_data= sorted(cleaned_data,key=lambda x: x[2])

    while len(cleaned_data)>originallen*.9:
   
        cleaned_data.pop()
    return cleaned_data
    
cleaned_data=outlierCleaner(predictions,features,labels)

import numpy
if len(cleaned_data) > 0:
    features, labels, errors = zip(*cleaned_data)

   


### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing



### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

from sklearn import linear_model, decomposition, datasets

from sklearn.neighbors import KNeighborsClassifier


selector=SelectKBest(k=18)
features_new=selector.fit_transform(features,labels)
#features=selector.transform(features)
#print features
print "help",selector.get_support()

clf=KNeighborsClassifier(weights='distance',leaf_size=11,metric='rogerstanimoto',p=1,n_neighbors=4)                    
 
clf.fit(features_new,labels)               


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

test_classifier(clf, my_dataset, features_list)

### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.

dump_classifier_and_data(clf, my_dataset, features_list)