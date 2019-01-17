import pandas as pd;

#finkcija za vcituvanje na podatocite od .csv file
def loadCsb(filename):
    data=pd.read_csv(filename,encoding="UTF-8");

    return data.values;
    #for row in reader:
     #   if(num_counter==0):
      #      print("the columns are"+str(row));
       # else:
        #    print(row);
        #num_counter+=1;





#finkcija koja vrakja recnik vo koi klucevi se klasite, i tie se mapirani vo lista od instancite(vleznite vektori pretstaveni so lista) koi se klasificirani so ovaa klasa
def separateByClass(inputVectors):
    #we will create a dictionary where every class will be mapped into a list of instances that have that class
    dictionary=dict();
    for i in range(1,len(inputVectors)):#kolku redovi imame
        classLabel=inputVectors[i][0]#klasata na i-tata instanca
        if(dictionary.get(classLabel)==None):
            dictionary[classLabel]=list();

        dictionary[classLabel].append(inputVectors[i]);
    return dictionary;


#laplacovo izramnuvanje na vo odnos na atributot so reden broj 'col' vo kontekst na 'key' klasata
def laplace(dataset,feature,col):

    from collections import Counter
    discreteDict = dict()


        #za listata od instanci od sekoja klasa
    for row in range(1,len(dataset)):
        #za sekoja instanca vo ovaa lista
        if ((dataset[row][col]) not in discreteDict.keys()):
            discreteDict[(dataset[row][col])] = 1
            #dodavanje po edno pojavuvanje za sekoja razlicna vrednost koja se srekjava vo trening podatocite, toest kreiranje na virtuelen primerok
            #so golemina kolku sto ima razlicni vrednosti i verojatnost p=1/(broj na razlicni vrednosti)

        continue

    temp = Counter(feature)
    #dodavanje na vistinskite pojavuvanja na vrednostite na ovoj atribut vo kontekst na klasata za koja se pravi izramnuvanje
    for tempkey in temp.keys():
        discreteDict[tempkey] += int(temp[tempkey])
    return discreteDict


#presmetka na statistikite za sekoja od klasite posebno
def summaryByClass(dictionary,dataset):
    import statistics as st
    from collections import Counter
    dictionaryWithStatistics=dict()

    #za sekoja klasa od trening podatocite
    for key in dictionary.keys():
        list=dictionary[key]
        #list se instancite odnosno vleznite vektori koi se od klasa 'key'
        dictionaryForCol=dict()
        #dictionaryForCol se koristi vo slucaj za da se mapira kolonata 'col' vo recnik od  sekoja razlicna vrednost na ovoj atributot mapiran vo brojot na nejzini pojavuvanja vo kontekst na klasata 'key'
        #ili dokolku atributot ne e diskreten, 'col' se mapira vo lista kade prviot element e aritmetickata sredina, vtoriot e standardnata devijacija od vrednostite na instancite za kolonata 'col'
        #print(list)
        for col in range(1,len(list[0])):
            #za site koloni vo podatocnoto mnozestvo
            feature = []
            #vo feature kje gi smestuvame site vrednosti na instancite za atributot koj soodvestvuva na kolonata so reden broj 'col' od klasata 'key'
            for row in list:
                feature.append(row[col])
                #gi zemame karakteristikite za kolonata col
            if(col==1 or col==3):
                #col 1 i 3 imaat neprekinati vrednosti
                floatFeature=[float(x) for x in feature]
                mu=round(st.mean(floatFeature),3);
                #presmetka na aritmeticka sredina na vrednostite od kolonata 'col' vo kontekst na klasata 'key'
                dictionaryForCol[col]=[mu,round(st.stdev(floatFeature,mu),3)]
            elif(col==2 or col==4):
                #discrete variable
                #Laplasovo izramnuvanje
                #Treba da proverime dali ima potreba od laplasovo izramuvanje
                lapl=False
                for k1 in dictionary.keys():
                    for row1 in dictionary[k1]:
                        if row1[col] not in feature:
                            #dokolku postoi barem edna vrednost za atributot od kolonata 'col' koj postoi vo celiot dataset a ne se pojavuva vo kontekst na klasata 'key'
                            #togas ima potreba od Laplasovo izramnuvanje
                            dictionaryForCol[col]=laplace(dataset,feature,col)
                            lapl=True
                            break
                        continue
                if not lapl:
                    dictionaryForCol[col]=Counter(feature)
        #se mapira sekoja klasa so statistikite za nea(recnik so soodvetnite statistiki za sekoja kolona)
        dictionaryWithStatistics[key]=dictionaryForCol
    return dictionaryWithStatistics

def classifier(dictionaryStatistics,instance,separatedByC,numberOfRows):
    probabilities=dict()
    import math as m
    #funkcija za presmetka na verojatnosta za vrednost na neprekinata slucajna promenliva(kolona so neprekinati vrednosti i pod pretpostavka deka se izvleceni od Gausova distribucija)
    normal= lambda x,mu,std:(1/(m.sqrt(2*m.pi)*std))*m.pow(m.e,(-m.pow((x-mu),2)/2*m.pow(std,2)))

    for key in dictionaryStatistics.keys():
        #za sekoja klasa vo trening podatocite
        prior = (len(separatedByC[key]) / numberOfRows)
        probability=1
        for col in dictionaryStatistics[key].keys():
            #za sekoja kolona toest za sekoj atribut vo trening podatocite
            if(col==1 or col==3):

                mu,std=dictionaryStatistics[key][col]
                #sredinata i  standardnata devijacija na vrednostite od momentalnata kolona(col) i vo kontekst na momentalnata klasa(key)
                prb=normal(float(instance[col-1]),mu,std)


                probability*=prb
                #presmetka na p(x|C)
                #x=float(instance[col-1])
                #col-1 bidejki vo ciklusot col pocnuva od 1, a vo instance vrednostite se pocnuvaat od indeks 0
            else:

                valueOfAtribute=instance[col-1]
                #vrednosta na atributot
                #Vo procesot na podgotovka na trening podatocite proverivme  dali vrednosta na ovoj atribut voopsto se srekjava vo trening podatocite koi pripagjaat na momentalnata klasa
                #dokolku e toj slucaj, primenivme laplasovo izramnuvanje
                prb=dictionaryStatistics[key][col][valueOfAtribute]/(sum([dictionaryStatistics[key][col][x] for x in dictionaryStatistics[key][col].keys()]))
                probability*=prb
                #presmetuvame p(x|C)=broj na instanci od trening podatocite kade x vrednosta se pojavila vo kontekst na C/vkupen broj na instanci koi se klasificirani so C
                #C klasate vo nasiot slucaj e 'key'
        probabilities[key]=probability*prior
        #posterior verojatnosta p(C|X) ja zapisuvame vo recnik
    return probabilities



def printStatistics(statistics):
    for key in statisticsByClass.keys():
        print("Class "+str(key))
        columns=len(statisticsByClass[key].keys())+1;
        for col in range(1,columns):
            if(col==1 or col==3):
                mean,std=statisticsByClass[key][col]
                print("The mean for column "+ str(col)+" is "+str(mean)+" and the standard deviation is "+str(std)+" ")

            else:
                for k in statisticsByClass[key][col].keys():
                    print("Appearance of "+str(k)+" "+str(statisticsByClass[key][col][k]))
                #print(statisticsByClass[key][col])
        print("\n");
if __name__=="__main__":
    filename='NaiveBayesTrainnigData.csv';
    dataset=loadCsb(filename);

    separatedByClassData=separateByClass(dataset)
    statisticsByClass=summaryByClass(separatedByClassData,dataset)
    print(statisticsByClass)


    probablilities=classifier(statisticsByClass,['3',"средно стручно",'8',"повеќе од 5"],separatedByClassData,len(dataset))
    #dobivame recnik kade sekoj klasa e mapirana vo verojatnosta deka test primerot e od taa klasa
    sum=sum(probablilities.values())
    #sum e p(x)
    for val in probablilities.values():
        print(val*100/sum)
    maxProb=0
    suggestedClass=None
    #ja naogjame klasata so najgolema posterior verojatnost odnosno p(C|x)
    for key in probablilities.keys():
        if(maxProb<probablilities[key]):
            maxProb=probablilities[key]
            suggestedClass=key

    print("The instance is aproximately "+str(maxProb*100/sum)+" percents frrom the class "+suggestedClass)
    #for dict in dictionaries:
     #   print(dict)

    #for i in range(len(inputVectors[0])):
     #   for j in range(len(inputVectors)):
      #      print(str(inputVectors[j][i])+" ",end="",flush=True)
       # print("\n");

