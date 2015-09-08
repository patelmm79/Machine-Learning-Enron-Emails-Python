#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees', 'to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi',  'shared_receipt_with_poi'] # You will need to use more features

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
        #except IndexError:

  
    while len(cleaned_data)>originallen*.9:
     
        cleaned_data.pop()
    return cleaned_data
    
cleaned_data=outlierCleaner(predictions,features,labels)

import numpy
if len(cleaned_data) > 0:
    features, labels, errors = zip(*cleaned_data)
   


### Task 3: Create new feature(s)

####MMP:  see PCA test below
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
#data = featureFormat(my_dataset, features_list, sort_keys = True)
#labels, features = targetFeatureSplit(data)


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn import linear_model, decomposition, datasets
from sklearn.grid_search import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import SelectKBest
decisiontree=tree.DecisionTreeClassifier()

from sklearn.neighbors import NearestNeighbors
pca = decomposition.PCA()
svc=SVC()
### MMP: select k-best features and print feature scores & best features

selector=SelectKBest(k=18)
features_new=selector.fit_transform(features,labels)

print  selector.scores_
#features=selector.transform(features)
#print features
print "help",selector.get_support()



###MMP: test decision tree


print "******Testing Decision Tree*******"


clf=decisiontree
clf.fit(features,labels)           
test_classifier(clf, my_dataset, features_list)    



###MMP: test Naive Bayes


print "******Testing Naive Bayes*******"


clf=GaussianNB()

clf.fit(features,labels)           
test_classifier(clf, my_dataset, features_list)    



###MMP: test K-Neighbors

print "******Testing K-Neighbors*******"

clf=KNeighborsClassifier()
clf.fit(features,labels)           
test_classifier(clf, my_dataset, features_list)    

###MMP: test PCA with K-neighbors


print "******Testing PCA with K-Neighbors*******"
n_components=[2,5,7,10,15]

parameters_to_tune = {'pca__n_components':n_components}

pipe = Pipeline(steps=[('pca', pca), ('neighbors',KNeighborsClassifier())])
estimator = GridSearchCV(pipe,parameters_to_tune,refit=True,scoring='recall_macro')
clf=estimator.fit(features,labels)
   
test_classifier(clf, my_dataset, features_list)  
print "best estimators are", clf.best_estimator_  

####MMP: Optimize K-Neighbors


print "******Optimising K-Neighbors*******"



parameters_to_tune = {#'pca__n_components':n_components,
                      'kbest__k':[2,5,10,15,'all'],
                          
                        'neighbors__n_neighbors': [2,3,4],
                
                      'neighbors__weights': ['distance'],
                      'neighbors__leaf_size':[10,11,15,30],
                       # 'neighbors__leaf_size':[15],
                    # 'neighbors__leaf_size':[30],
                     'neighbors__metric':['euclidean', 'manhattan', 'chebyshev','minkowski', 'wminkowski','seuclidean','mahalanobis','haversine','hamming', 'canberra', 'braycurtis','jaccard','maching','dice','kulsinski', 'rogerstanimoto', 'russellrao','sokalmichener','sokalsneath'],
                    # 'neighbors__metric':['canberra', 'braycurtis','jaccard','dice','kulsinski', 'rogerstanimoto', 'russellrao','sokalmichener','sokalsneath'],#,'haversine','hamming'],
                    #'neighbors__metric':['rogerstanimoto'],
                    'neighbors__algorithm' : ['auto'],
                    'neighbors__p':[1,2,3]
                   
                   #   'neighbors__algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute']
                     ## 'decisiontree__max_features':[None, 2, 4, 6, 8],
                     }

pipe = Pipeline(steps=[('kbest', SelectKBest(f_classif)),('neighbors',KNeighborsClassifier())])

clf=estimator.fit(features,labels)


print "best estimators are", clf.best_estimator_

       
test_classifier(clf, my_dataset, features_list)    






### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

test_classifier(clf, my_dataset, features_list)

### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.

#dump_classifier_and_data(clf, my_dataset, features_list)