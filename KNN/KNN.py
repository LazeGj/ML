import pandas as pd
import numpy as np
import statistics as st
from sklearn.preprocessing import StandardScaler

def loadData(filename):
    data=pd.read_csv(filename,names=['A'+str(x) for x in range(1,17)])

    dataWithNoMissings=data.replace(['?',' '],np.NaN)
    dataWithNoMissings.dropna(inplace=True)



    '''
    dataToProces=data.loc[:,['A'+str(x) for x in range(1,16)]].values
    clases=data.loc[:,'A16']
    #dealing with missing values
    #missing values
    #A1: 12
    #A2: 12
    #A4: 6
    #A5: 6
    #A6: 9
    #A7: 9
    #A14: 13
    atr=[i for i in range(0,15)]
    dataWithNoMissingV=[]
    for row  in dataToProces:
        flag=True
        for i in atr:

            if(row[i]=='?' or row[i]==' '):
                flag=False
                break
        if(flag):
            dataWithNoMissingV.append(row)
    #Now we should deal with the category atributes
    #We will use the One-Hot method to transform the non-numerical atribtues to numerical
    dataWithNoMissingV=pd.DataFrame(data=dataWithNoMissingV,columns=['A'+str(x) for x in range(1,16)])
    dataWithNoMissingV['A2']=[float(x) for x in dataWithNoMissingV['A2']]
    dataWithNoMissingV['A14']=[float(x) for x in dataWithNoMissingV['A14']]
'''

    print(dataWithNoMissings.shape)

    #finding the atributes(columns) that have  not numerical values
    columns=[]
    for atribute in dataWithNoMissings.keys():
        if(atribute=='A16'):
            continue
        if(atribute not in ['A2','A3','A8','A11','A14','A15']):
            dataWithNoMissings[atribute].astype('category')
            columns.append(atribute)

    #the non numerical data is converted to numerical using one-hot method
    numericalData=pd.get_dummies(dataWithNoMissings,columns=columns,prefix=['o-hot' for x in columns])


    return numericalData

#This function is for data normalization
def normaliseData(dataFrame):

    continousFeatures=[x for x in ['A2','A3','A8','A11','A14','A15']]
    #separating the features

    x=dataFrame.loc[:,continousFeatures]
    x=pd.DataFrame(data=StandardScaler().fit_transform(x),columns=continousFeatures)

    for feature in continousFeatures:
        dataFrame[feature]=x[feature]
    return dataFrame

def dimensionalityReduction(dataFrame):
    from sklearn.decomposition import PCA

    pca=PCA(n_components=3)
    data_frame.dropna(inplace=True)
    principalComponentData=pca.fit_transform(dataFrame)
    principalDf=pd.DataFrame(data=principalComponentData,columns=['principalC1','principalC2','principalC3'])
    return principalDf

def showingDataPlot(dataFrame):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111, projection='3d')
    zline = dataFrame['principalC3']
    xline = dataFrame['principalC1']
    yline = dataFrame['principalC2']
    #ax.scatter(xline, yline, zline, c='b')
    ax.set_title('Credit aproval data set 3d with PCA',loc='left')

    ax.margins(x=-0.1,y=0.1,z=0)
    #i=0

    ax.set_xlabel('principalC1')
    ax.set_ylabel('principalC2')
    ax.set_zlabel('principalC3')



    classes=dataFrame['class']


    ax.scatter(xline[0], yline[0], zline[0], c='b', label='Positive-the credit was approved')
    ax.scatter(xline[652], yline[652], zline[652], c='r', label='Negative-the credit was not approved')

    i=1

    for c in classes:
       if(c=='+'):
           ax.scatter(xline[i], yline[i], zline[i], c='b')
       else:
           ax.scatter(xline[i], yline[i], zline[i], c='r')
       i+=1
       if(i==652):
           break



    ax.legend(loc='best')
    plt.show()

def creatingTestSet(data_frame):
    testSetIndexes=[]
    test_set=[]
    instances=data_frame.values
    n=len(instances)
    test_size=int(n*0.3)
    import random
    random.seed()
    while(len(test_set)<test_size):
        p=random.randrange(0,n,1)
        instance=instances[p,:].tolist()
        if( p not in testSetIndexes):
            test_set.append(instance)
            testSetIndexes.append(p)
    return test_set,testSetIndexes


def EuclidDistance(vector1,vector2):
    import math
    sum=0;
    for i in range(0,len(vector1)-1):
        sum+=pow((vector1[i]-vector2[i]),2)

    return math.sqrt(sum)

def ManhattanDistance(vector1,vector2):
    import math
    sum=0;
    for i in range(0,len(vector1)-1):
        sum+=abs((vector1[i]-vector2[i]))

    return sum

class KNNclassifier:
    def __init__(self,Kneighbours):
        self.K=Kneighbours


    def classify(self,test_instance,data_set,numberClass,distanceMetric=EuclidDistance):
        #data_set i list of input vectors
        Knearest=[]

        distanceDict=dict()
        max=0
        maxVector=None
        size=len(data_set)
        for vector in range(0,size):
            distance=distanceMetric(test_instance,data_set[vector])
            if(len(Knearest)<int(self.K)):
                #Every instance is mapped to its distance from the test_instance based on the given distance metric
                distanceDict[vector]=distance
                if(max<distanceDict[vector]):
                    max=distanceDict[vector]
                    maxVector=vector
                    #If we have currently the maximum distance between to the test_instace from the K nearest neighbours
                    #then for every new vector we just need to check if its distance from the test_instace is greater max,
                    # if it is then there is no need to compare him with the other K kearest neighbours
                Knearest.append(vector)
            elif(distance>=max):
                continue


            else:
                Knearest.remove(maxVector)
                Knearest.append(vector)
                distanceDict[vector]=distance
                max=0
                for k in Knearest:
                    if(distanceDict[k]>max):
                        max=distanceDict[k]
                        maxVector=k


        #we will clasify the test instance based on the majority in Knearest
        positive=0
        for k in Knearest:
            if(data_set[k][numberClass]=='+'):
                positive+=1
        if(positive>len(Knearest)-positive):
            return '+'
        return '-';


    def classify_test_set(self,test_set,data_set,classNumber,distanceMetric):
        classified=dict()
        n=len(test_set)
        for row in range(0,n):
            c=self.classify(data_set[row],data_set,classNumber,distanceMetric)
            classified[row]=c
        return classified


def calculateTheMistakes(K,test_set,data_set,distanceMetric,classNumber):
    classifier = KNNclassifier(K)
    tested = classifier.classify_test_set(test_set, data_set, classNumber, distanceMetric)
    mistakes = 0
    for t in tested.keys():
        if (not data_set[t][classNumber] == tested[t]):
            mistakes += 1

    return mistakes


def crossValidation(Kvalues,data_set,test_set,classNumber):
    performanceEuclidean=dict()
    performanceManh=dict()

    #with Euclidean distance
    for k in Kvalues:

        performanceEuclidean[k]=calculateTheMistakes(k,test_set,data_set,EuclidDistance,classNumber)

    for k in Kvalues:

        performanceManh[k]=calculateTheMistakes(k,test_set,data_set,ManhattanDistance,classNumber)


    #plot the performance

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ks=[x for x in performanceManh.keys()]
    ysWithEuclidiean=[performanceEuclidean[x] for x in performanceEuclidean.keys()]
    ysWithManh = [performanceManh[x] for x in performanceManh.keys()]

    ax.plot(ks,ysWithEuclidiean,c='r',label='With Euclidian distance')
    ax.plot(ks,ysWithManh,c='b',label='With Manhattan distance')
    plt.xticks(range(2,16))

    ax.set_xlabel('Neighbours')
    ax.set_ylabel("Mistakes on the test set(195 instances)")
    ax.legend(loc='best')
    plt.show()

    #Now to return the number of neighbours and the distance metric for which the KNN algorithm gives best performance

    min_mistakes=performanceEuclidean[2]
    min_k=2
    metric=EuclidDistance
    for k in performanceEuclidean.keys():
        if(performanceEuclidean[k]<min_mistakes):
                min_mistakes=performanceEuclidean[k]
                min_k=k


    for k in performanceManh.keys():
        if(performanceManh[k]<min_mistakes):
            min_mistakes=performanceManh[k]
            min_k=k
            metric=ManhattanDistance

    return min_k,metric




if __name__=='__main__':
    data_frame=loadData('CreditAproval.csv')
    #save the proccesed data
    data_frame.rename(columns={"A16":"class"},inplace=True)
    data_frame.to_csv('CreditAprovalPrepared.csv')
    #data_frame=pd.read_csv('CreditAprovalPrepared.csv')
    #classes=data_frame.loc[:,'class']
    #data_frame=data_frame.drop(columns=['class','Unnamed: 0'])
    print(data_frame.isnull().sum().sum())
    normalizedData=normaliseData(data_frame)
    classes=normalizedData['class']
    normalizedData=normalizedData.drop(columns=["class"])
    normalizedData.dropna(inplace=True)
    #print(normalizedData.isnull().sum().sum())

    principalComponent3D=dimensionalityReduction(normalizedData)

    #Normalize PCA values
    principalComponent3D=pd.DataFrame(data=StandardScaler()
.fit_transform(principalComponent3D.values),
columns=principalComponent3D.keys())
    principalComponent3D['class']=classes


    test_set,test_set_indexes=creatingTestSet(principalComponent3D)

    #da se otstranat onie instance koi bea izbrani za testiranje
    trainningData = principalComponent3D.drop(test_set_indexes)



    #crossValidation
    best_k, best_metric = crossValidation([x for x in range(2, 15)], trainningData.values.tolist(),
                                          test_set, 3)


    #performance with the best parametars
    mistakes=calculateTheMistakes(best_k,test_set,trainningData.values.tolist(),best_metric,3)
    print(((len(test_set)-mistakes)/len(test_set))*100)
    #print(len(test_set))
    #test_set=pd.DataFrame(data=test_set,columns=[x for x in principalComponent3D.keys()],index=test_set_indexes)
    #test_set.to_csv('TestSetKNN.csv')










