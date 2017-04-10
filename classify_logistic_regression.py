from __future__ import division
import unicodecsv
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek

## Read csv file
train = pd.read_csv('train.csv', parse_dates = ['Dates'])
test = pd.read_csv('test.csv', parse_dates = ['Dates'])

## Manipulate with origin data
def maniplt_with_origin_data(data):
    DayOfWeek_dict = {'Sunday':0,'Monday':1,'Tuesday':2,'Wednesday':3,'Thursday':4,'Friday':5,'Saturday':6}
    data['DayOfWeek'] = data['DayOfWeek'].map(lambda x: DayOfWeek_dict[x])

    ## PdDistrict
    le_PdDistrict = LabelEncoder()
    data['PdDistrict'] = le_PdDistrict.fit_transform(data['PdDistrict'])

    ## Dates
    data['Year'] = data['Dates'].map(lambda x: x.year)
    data['Month'] = data['Dates'].map(lambda x: x.month)
    # data['Day'] = data['Dates'].map(lambda x: x.day)
    data['Hour'] = data['Dates'].map(lambda x: x.hour)
    # data['Minute'] = data['Dates'].map(lambda x: x.minute)

    data['time'] = data['Hour'].apply(lambda x: 1 if (x >= 1 and x <= 7)
                                                else 2 if (x >= 8 and x <= 15)
                                                else 3)
    data['season'] = data['Month'].apply(lambda x: 1 if (x >= 3 and x <= 5)
                                                else 2 if (x >= 6 and x <= 8)
                                                else 3 if (x >= 9 and x <= 11)
                                                else 4)
    
    ## Address
    data['StrNo'] = data['Address'].apply(lambda x: x.split(' ', 1)[0] if x.split(' ', 1)[0].isdigit() else 0)
    data["Inter"]= data["Address"].apply(lambda x: 1 if "/" in x else 0)
    data['Address'] = data['Address'].apply(lambda x: x.split(' ', 1)[1] if x.split(' ', 1)[0].isdigit() else x)
    le_Address = LabelEncoder()
    data['Address'] = le_Address.fit_transform(data['Address'])

    ## X, Y
    data['X'] = data['X'].round(3)
    data['Y'] = data['Y'].round(3)
    return data

## Training Set
## Convert the features to numerical values
featuers_train = maniplt_with_origin_data(train)
featuers_train.drop(featuers_train[featuers_train.Y == 90].index, inplace=True)
le_Category = LabelEncoder()
labels_train = le_Category.fit_transform(featuers_train['Category'])
# featuers_train.drop(['Category', 'Descript', 'Resolution', 'Dates'], axis=1, inplace=True)
featuers_train.drop(['Category', 'Descript', 'Resolution','Dates','Month','Hour'], axis=1, inplace=True)
# sm = SMOTE(random_state=42)
# featuers_train, labels_train = sm.fit_sample(featuers_train, labels_train)

## Testing Set
featuers_test = maniplt_with_origin_data(test)
# featuers_test.drop(['Id','Dates'], axis=1, inplace=True)
featuers_test.drop(['Id','Dates','Month','Hour'], axis=1, inplace=True)

## Missing Values
# print featuers_train.isnull().sum(axis=0)

## Normalization
# featuers_train = StandardScaler().fit_transform(featuers_train)

# ## PCA
# pca = PCA(n_components='mle', whiten=True, svd_solver='full').fit(featuers_train)
# pca_num = pca.n_components_
# ipca = PCA(n_components=pca_num, whiten=True).fit(featuers_train)
# ipca.transform(featuers_train)
# ipca.transform(featuers_test)

## Implement Logistic Regression
clf = LogisticRegression()
clf.fit(featuers_train, labels_train)
pred = clf.predict_proba(featuers_test)

## Export test set
result = pd.DataFrame(pred,columns=le_Category.classes_)
result.to_csv("logistic_regression.csv", index = True, index_label = "Id")
