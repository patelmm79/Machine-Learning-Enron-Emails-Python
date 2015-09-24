#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Imputer
from numpy import column_stack
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

features_list = ['poi','salary','deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees', 'to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi',  'shared_receipt_with_poi'] # You will need to use more features
#features_list = ['poi','total_payments', 'loan_advances', 'bonus','rest]
#features_list = ['poi','bonus','total_payments','loan_advances']
#'from_poi_to_this_person',
#'from_this_person_to_poi',
#'shared_receipt_with_poi',
 # You will need to use more features

#features_list = ['poi','salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees', 'to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi',  'shared_receipt_with_poi'] # You will need to use more features
### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )
my_dataset = data_dict


### Task 2: Remove outliers
del data_dict['TOTAL']
del data_dict['THE TRAVEL AGENCY IN THE PARK']
del data_dict['LOCKHART EUGENE E']









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


##features_list.append('from_poi_ratio')
features_list.append('to_poi_ratio')
features_list.remove('from_poi_to_this_person')
#features_list.remove('from_this_person_to_poi')
features_list.remove('to_messages')
#features_list.remove('salary')
#features_list.remove('from_messages')


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



### Store to my_dataset for easy export below.

### Extract features and labels from dataset for local testing



### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html







data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)



from sklearn import linear_model, decomposition, datasets

from sklearn.neighbors import KNeighborsClassifier


selector=SelectKBest(k=15)
#rint labels
features_new=selector.fit_transform(features,labels)


print features_list
##print features_new
print  selector.scores_
features=selector.transform(features)
#print features
print "help",selector.get_support()

#clf= Pipeline(steps=[('kbest', SelectKBest(k='all')),('neighbors',KNeighborsClassifier(weights='distance',leaf_size=2,metric='jaccard',p=1,n_neighbors=4))])

######this is the best w/o new features
#clf= Pipeline(steps=[('kbest', SelectKBest(k='all')),('neighbors',KNeighborsClassifier(weights='distance',leaf_size=2,metric='rogerstanimoto',p=1,n_neighbors=4))])


clf= Pipeline(steps=[('kbest', SelectKBest(k='all')),('neighbors',KNeighborsClassifier(weights='distance',leaf_size=2,metric='rogerstanimoto',p=1,n_neighbors=4))])


#clf= Pipeline(steps=[('kbest', SelectKBest(k='all')),('neighbors',KNeighborsClassifier(weights='distance',leaf_size=3,metric='jaccard',p=1,n_neighbors=3))])

#clf=KNeighborsClassifier(weights='distance',leaf_size=1,metric='rogerstanimoto',p=1,n_neighbors=4)
#clf=KNeighborsClassifier(weights='distance',leaf_size=1,metric='jaccard',p=1,n_neighbors=3)
clf.fit(features_scaled,labels)

#clf.fit(features_new,labels)      
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

test_classifier(clf, my_dataset, features_list)

### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.

dump_classifier_and_data(clf, my_dataset, features_list)