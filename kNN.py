import unicodecsv
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.metrics import log_loss

## open file
train = pd.read_csv('train.csv', parse_dates = ['Dates'])
test = pd.read_csv('test.csv', parse_dates = ['Dates'])

## manipulate with origin data
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
# featuers_train.drop(['Category', 'Descript', 'Resolution','Dates'], axis=1, inplace=True)
featuers_train.drop(['Category', 'Descript', 'Resolution','Dates','Month','Hour'], axis=1, inplace=True)
# sm = SMOTE(random_state=42)
# featuers_train, labels_train = sm.fit_sample(featuers_train, labels_train)

# # Testing Set
# featuers_test = maniplt_with_origin_data(test)
# featuers_test.drop(['Id','Dates'], axis=1, inplace=True)

## Missing Values
# print featuers_train.isnull().sum(axis=0)

## Normalization
# featuers_train = StandardScaler().fit_transform(featuers_train)

## Split the training set into training and validation parts
X_train, X_test, y_train, y_test = train_test_split(featuers_train, labels_train, test_size=0.3, random_state=1)


## PCA
# pca = PCA(n_components='mle', whiten=True, svd_solver='full').fit(X_train)
# pca_num = pca.n_components_
# ipca = PCA(n_components=pca_num, whiten=True).fit(X_train)
# ipca.transform(X_train)
# ipca.transform(X_test)

## Implement kNN
clf = KNeighborsClassifier(n_neighbors=200)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, pred)
print "The accuracy is", accuracy



# K = range(100, 2000, 400)
# accuracy_list = []
# for k in K:
#     clf = KNeighborsClassifier(n_neighbors=k)
#     clf.fit(X_train, y_train)
#     pred = clf.predict(X_test)
#     accuracy_list.append(accuracy_score(y_test, pred))
# plt.plot(K, accuracy_list, 'bx-')
# plt.xlabel('k')
# plt.ylabel('Accuracy')
# plt.title('Selecting k with the Elbow Method')
# plt.show()