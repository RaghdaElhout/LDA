import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix


#READ DATA
data = pd.read_csv("data.csv")

data.drop(data.columns[[-1, 0]], axis=1, inplace=True)
data['diagnosis'].replace('B',0,inplace=True)
data['diagnosis'].replace('M',1,inplace=True)



#SPLIT INTO TRAIN AND TEST
X, y = data.iloc[:, 1:].values, data.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

#STANDRIZE DATA
scaler = StandardScaler()

# Fit on training set only.
scaler.fit(X_train)

train_std = scaler.transform(X_train)
test_std = scaler.transform(X_test)


#PCA TO CAPTURE 95% of VARIANCE IN DATA
pca = PCA(.95)

pca.fit(train_std)

X_train = pca.transform(train_std)
X_test = pca.transform(test_std)




#VISUALIZING DATA USING FIRST 2 PCA
principalDf = pd.DataFrame(data = -1*X_train[:,0:2], columns = ['principal component 1', 'principal component 2'])
PCALables=pd.DataFrame(data = y_train, columns = ['Labels'])
finalDf = pd.concat([principalDf, PCALables], axis = 1)



fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = [0, 1]
colors = ['r','b']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['Labels'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid(True)
plt.show()



#CLASSIFICATION USING LDA AND ROC-AUC

lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
score = lda.score(X_test, y_test)
predections = lda.predict(X_test)
prob=lda.predict_proba(X_test)
auc = roc_auc_score(y_test, predections)

fpr, tpr, thresholds = metrics.roc_curve(y_test, prob[:,1])
plt.plot(fpr, tpr)
plt.show()


print "AUC: ",auc
print "Accuracy= ",(score)


#CALCULATE CONFUSION MATRIX
conf_mat = confusion_matrix(y_test, predections)
print "Confusion matrix: ", conf_mat
