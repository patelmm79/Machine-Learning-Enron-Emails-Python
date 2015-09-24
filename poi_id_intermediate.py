#!/usr/bin/python

import sys
import pickle
import numpy
from pandas import DataFrame
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Imputer
from numpy import column_stack

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees', 'to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi',  'shared_receipt_with_poi'] # You will need to use more features

### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

print "wtf"
            
### Task 2: Remove outliers

            
#####MMP: check names and number of blanks           
            
            
            
NanDict={}


for key in data_dict.keys():
    NanDict[key]=0
    for feature in features_list:
            if data_dict[key][feature]=='NaN':
                NanDict[key]=NanDict[key]+1
print "Summary dictionary of keys and missing values:"      
print NanDict

###Two obvious items should not be in the list: 'TOTAL', and 'THE TRAVEL AGENCY IN THE PARK'

del data_dict['TOTAL']
del data_dict['THE TRAVEL AGENCY IN THE PARK']
del NanDict['THE TRAVEL AGENCY IN THE PARK']
del NanDict['TOTAL']


###How many keys are missing  a large number of data points?
print "Keys with high number of missing data points:"
print {k: v for k, v in NanDict.items() if v > 15}

###One  key, "LOCKHART EUGENE E" , has 19 values not available.  let's remove him as well.

del data_dict['LOCKHART EUGENE E']



#print data_dict
overview=DataFrame.from_dict(data_dict, orient='index')

print "======="

print "here?"
   


### Task 3: Create new feature(s)

for key in data_dict.keys():
    if data_dict[key]['from_poi_to_this_person']=='NaN' or data_dict[key]['from_messages']== 'NaN':
            data_dict[key]['from_poi_ratio']='NaN'
            
           
    else:
        
            data_dict[key]['from_poi_ratio']=float(data_dict[key]['from_poi_to_this_person'])/ float(data_dict[key]['from_messages'])
    if data_dict[key]['from_this_person_to_poi']=='NaN' or data_dict[key]['to_messages']== 'NaN':
    
        data_dict[key]['to_poi_ratio']='NaN'
    else:
        data_dict[key]['to_poi_ratio']=float(data_dict[key]['from_this_person_to_poi'])/ float(data_dict[key]['to_messages'])


#features_list.append('from_poi_ratio')
features_list.append('to_poi_ratio')

print "here?"

def rescale_process(feature):
    rescaled_array=[]
    label=[]
    scaler=MinMaxScaler(feature_range=(0, 1))
    
    if  feature=='poi':    
        for key in data_dict.keys():
    
            if data_dict[key][feature]==True:
                
                label.append(1)
            else:
                label.append(0)
        
        return label
      
    else:
        for key in data_dict.keys():
            rescaled_array.append(float(data_dict[key][feature]))
        imp = Imputer(missing_values='NaN', strategy='mean', axis=1)
       
 
        array_imputed=imp.fit_transform(rescaled_array)
    
        array_rescaled=scaler.fit_transform(array_imputed[0])
   
  
        return array_rescaled
scaled_array=[]
labels_array = []
for feature in features_list:
    if feature !='poi':
        #scaled_array.append(numpy.array((rescale_process(feature))))
        scaled_array.append(rescale_process(feature))

            
    else:
       labels_array=rescale_process(feature)
labels_scaled=labels_array
features_scaled=column_stack(scaled_array)

my_dataset = data_dict

#for item in features_list:
    



### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


#print "labels are scaled, son", len(features_scaled[1])
#print "features=", features
print "length of features", len(features)
print "number of POIs:", len(overview[overview['poi']==True])



#print "number of POIs:", len(overview[overview['salary']=='NaN'])





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

#print features

#print labels
print  selector.scores_
#features=selector.transform(features)
#print features
print "help",selector.get_support()



###MMP: test decision tree


print "******Testing Decision Tree*******"


clf=decisiontree
#clf.fit(features,labels)           
#test_classifier(clf, my_dataset, features_list)    



###MMP: test Naive Bayes


print "******Testing Naive Bayes*******"


clf1=GaussianNB()

#clf1.fit(features,labels)           
#test_classifier(clf1, my_dataset, features_list)    



###MMP: test K-Neighbors

print "******Testing K-Neighbors*******"


print "K-Neighbors Scaled"


clf2=KNeighborsClassifier()
#features_what = scaled_array
clf2.fit(features_scaled,labels_scaled)
test_classifier(clf2, my_dataset, features_list)

print "K-Neighbors Unscaled"
 

clf3=KNeighborsClassifier()
clf3.fit(features,labels)
test_classifier(clf3, my_dataset, features_list)

###MMP: test PCA with K-neighbors


print "******Testing PCA with K-Neighbors*******"
n_components=[2,5,7,10,15]

parameters_to_tune = {'pca__n_components':n_components}

pipe = Pipeline(steps=[('pca', pca), ('neighbors',KNeighborsClassifier())])
#estimator = GridSearchCV(pipe,parameters_to_tune,refit=True,scoring='recall_macro')
#clf=estimator.fit(features,labels)
   
#test_classifier(clf, my_dataset, features_list)  
#print "best estimators are", clf.best_estimator_  

####MMP: Optimize K-Neighbors


print "******Optimising K-Neighbors*******"



parameters_to_tune = {#'pca__n_components':n_components,
                      'kbest__k':[17,18,19,20,'all'],
                          
                        'neighbors__n_neighbors': [2,3,4],
                
                      'neighbors__weights': ['distance'],
                      'neighbors__leaf_size':[1,5,10,11,15,30],
                       # 'neighbors__leaf_size':[15],
                    # 'neighbors__leaf_size':[30],
                     'neighbors__metric':['euclidean', 'manhattan', 'chebyshev','minkowski','hamming', 'canberra', 'braycurtis','jaccard','dice','kulsinski', 'rogerstanimoto', 'russellrao','sokalmichener','sokalsneath'],
                    # 'neighbors__metric':['canberra', 'braycurtis','jaccard','dice','kulsinski', 'rogerstanimoto', 'russellrao','sokalmichener','sokalsneath'],#,'haversine','hamming'],
                    #'neighbors__metric':['rogerstanimoto'],
                    'neighbors__algorithm' : ['auto'],
                    'neighbors__p':[1,2,3]
                   
                   #   'neighbors__algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute']
                     ## 'decisiontree__max_features':[None, 2, 4, 6, 8],
                     }

print "here?"
pipe = Pipeline(steps=[('kbest', SelectKBest(f_classif)),('neighbors',KNeighborsClassifier())])
estimator=GridSearchCV(pipe,parameters_to_tune, refit=True,scoring='recall_macro')



clf4=estimator.fit(features_scaled,labels)

test_classifier(clf4, my_dataset, features_list)    

print  selector.scores_


print "best estimators are", clf4.best_estimator_


print "******Testing optimised K-Neighbors*******"
clf5=KNeighborsClassifier(algorithm='auto',weights='distance',leaf_size=1,metric='chebyshev',p=1,n_neighbors=4)
#selector=SelectKBest(k=15)
features_new=selector.fit_transform(features_scaled,labels)
 
 
clf5.fit(features_new,labels)

       
test_classifier(clf5, my_dataset, features_list)






### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

#test_classifier(clf, my_dataset, features_list)

### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.

#dump_classifier_and_data(clf, my_dataset, features_list)