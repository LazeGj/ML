import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn.neural_network import MLPClassifier
from sklearn.metrics.classification import accuracy_score


from sklearn.svm import SVC

class Classifier:
    def __init__(self,fileData):
        self.fileData=fileData

    def loadData(self):
        data=pd.read_csv(self.fileData,delimiter=',',header=None,names=['Age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week'

                                                                      ,'native-country','class'])
        # data.describe()
        data=data.dropna()



        self.data=data


    def preprocessData(self):

        data=pd.DataFrame()
        for (type,key) in zip(self.data.dtypes,self.data.keys()):
            if(type=='object'):
                encoder=LabelEncoder()
                data[key]=encoder.fit_transform(self.data[key])
            else:
                data[key]=self.data[key]

        #encoder=LabelEncoder().fit(self.data[])
        #print(encoder)
        classLabel=data.pop('class')
        scaler=StandardScaler().fit(data)
        self.data=pd.DataFrame(data=scaler.fit_transform(data),columns=data.keys())

        self.data_labels=classLabel



    def createTestSet(self,testSetSize):

        self.data_train,self.data_test,\
        self.train_labels,self.test_labels=train_test_split(self.data,self.data_labels,test_size=testSetSize)



    def crossValidationParametarsNN(self):
        activations=['relu','logistic','tanh']
        alphas=[0.001,0.0001,0.0003,0.0009,
                0.003]

        #for every combination of activatin function and alhpa
        #the model will be trained for 10 iterations and as a performance will be taken the average performance of thoose 5 iter
        best_score=0
        best_activation=None
        best_alpha=None
        for activation in activations:

            for alpha in alphas:
                avg=0;
                for i in range(0, 5):
                    model=MLPClassifier(hidden_layer_sizes=(25,25),activation=activation,alpha=alpha,max_iter=500)
                    model.fit(self.data_train,self.train_labels)
                    pred_labels=model.predict(self.data_test)
                    score=accuracy_score(self.test_labels,pred_labels)
                    avg=avg+score
                avg=avg/10
                if(avg>best_score):
                    best_score=score
                    best_activation=activation
                    best_alpha=alpha

        return best_activation,best_alpha


    def crossValidationKNN(self):

        best_k=0
        best_score=0
        for i in range(2,10):
            knn = neighbors.KNeighborsClassifier(n_neighbors=5)
            knn.fit(self.data_train, self.train_labels)
            pred_labels = knn.predict(self.data_test)
            score=accuracy_score(self.test_labels,pred_labels)
            if(score>best_score):
                best_score=score
                best_k=i
        return best_k


    def cvModel(self):
        #choose the best model

        #model 1: Logistig regression
        logisticRegression = LogisticRegression(max_iter=1000)
        logisticRegression.fit(self.data,self.data_labels)
        print(logisticRegression.score(self.data_test,self.test_labels))
        #model 2:Neural networks

        activation,alpha=self.crossValidationParametarsNN()
        mlpClassifier=MLPClassifier(alpha=alpha,hidden_layer_sizes=(25,25),activation=activation,max_iter=500)
        mlpClassifier.fit(self.data_train,self.train_labels)
        pred_labels=mlpClassifier.predict(self.data_test)
        print(accuracy_score(self.test_labels,pred_labels))

        #model 3: SVM

        svc=SVC(kernel='rbf')
        svc.fit(self.data_train,self.train_labels)
        print(svc.score(self.data_test,self.test_labels))


        #model 4: KNN

        knn=neighbors.KNeighborsClassifier(n_neighbors=5)
        knn.fit(self.data_train,self.train_labels)
        pred_labels=knn.predict(self.data_test)
        print(accuracy_score(self.test_labels,pred_labels))



if __name__=='__main__':
    classifier=Classifier('Income.csv')
    classifier.loadData()
    classifier.preprocessData()
    classifier.createTestSet(0.2)
    classifier.cvModel()