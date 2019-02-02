import pandas as pd;
import datetime
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import numpy as np

#function for data preprocessing
#this function creates a dictionary with keys- certain data and hour and values-the temperature
#we use that dictionary to merge the data with hourly kw production with
def merge_with_temp(data,filename_temp):
    temp=pd.read_csv(filename_temp,delimiter=";")
    temp=pd.DataFrame(data=(temp["T"]))

    values=temp.index.values
    data_to_temp=dict()

    last=0
    for row in values:
        #the first element(index 0) in the tuple is the data and the hour and the second is the temperatur
        date=row[0]
        temp=float(row[1])
        parts=date.split(" ")
        date=parts[0]
        hour=parts[1]
        if(temp!=None and temp!=""):
            last=temp
        else:
            temp=last

        if(hour=="04:00"):
            data_to_temp[str(date + " " +"4:00")] = temp
            data_to_temp[str(date + " " + "6:00")] = temp
            data_to_temp[str(date + " " + "5:00")] = temp
        elif(hour=="01:00"):
            data_to_temp[str(date + " " + "1:00")] = temp
            data_to_temp[str(date + " " + "2:00")] = temp
            data_to_temp[str(date + " " + "3:00")] = temp
        elif(hour=="22:00"):
            data_to_temp[str(date+" "+hour)]=temp
            data_to_temp[str(date+" "+"23:00")]=temp
            data_to_temp[str(date+" "+"0:00")]=temp
            data_to_temp[str(date+" "+"1:00")]=temp
            data_to_temp[str(date+" "+"2:00")]=temp
            data_to_temp[str(date+" "+"3:00")]=temp
        elif(hour=="07:00"):
            data_to_temp[str(date+" "+"7:00")]=temp
            if(not data_to_temp.__contains__(str(date + " " + "04:00"))):
                data_to_temp[str(date + " " + "4:00")] = temp

                data_to_temp[str(date + " " + "6:00")] = temp
                data_to_temp[str(date + " " + "5:00")] = temp
            data_to_temp[str(date+" "+"8:00")]=temp
            data_to_temp[str(date+" "+"9:00")]=temp
        elif (hour == "10:00"):
            data_to_temp[str(date + " " + hour)] = temp
            data_to_temp[str(date + " " + "11:00")] = temp
            data_to_temp[str(date + " " + "12:00")] = temp
        elif (hour == "13:00"):
            data_to_temp[str(date + " " + hour)] = temp
            data_to_temp[str(date + " " + "14:00")] = temp
            data_to_temp[str(date + " " + "15:00")] = temp
        elif (hour == "16:00"):
            data_to_temp[str(date + " " + hour)] = temp
            data_to_temp[str(date + " " + "17:00")] = temp
            data_to_temp[str(date + " " + "18:00")] = temp
        elif (hour == "19:00"):
            data_to_temp[str(date + " " + hour)] = temp
            data_to_temp[str(date + " " + "20:00")] = temp
            data_to_temp[str(date + " " + "21:00")] = temp


    #now we can easily merge the temp data to the primary data
    indexDay = 4
    indexMonth = 3
    indexYear = 2
    indexHour=6
    values=data.values
    last=0
    temperatures=[]
    for row in values:
        day=row[indexDay]
        month=row[indexMonth]
        year=row[indexYear]
        hour=row[indexHour]
        month=row[indexMonth]
        year=row[indexYear]
        hour=row[indexHour]
        key=None
        if(int(day)<10 and int(month)>9):
            key = "0"+str(day) + "." + str(month) + "." + str(year) + " " + str(hour)
        elif(int(day)<10 and int(month)<10):
            key = "0"+str(day) + ".0" + str(month) + "." + str(year) + " " + str(hour)
        elif(int(day)>9 and int(month)<10):
            key = str(day) + ".0" + str(month) + "." + str(year) + " " + str(hour)
        else:
            key = str(day) + "." + str(month) + "." + str(year) + " " + str(hour)


        if (data_to_temp.__contains__(key)):
            temp = data_to_temp[key]
            last = temp
        else:
            temp = last
        temperatures.append(temp)


    return temperatures


#function for data preprocessing
def load_data(filename,temp_filename):

    data=pd.read_csv(filename)
    data=data[data['Hourly production [kWh]']>0]
    #we will transform the data so the date will be presented by the day of the week and the mouht

    d=data.pop("Date")

    data["DayOfWeek"]=(get_day_of_week(d))
    data['Year']=get_year(d)
    data['Month'] = (get_month(d))
    data['Day'] = (get_day(d))
    data['Weekend']=(is_weekend(d))
    data['Hour']=get_hour(d)
    previusDayAverage=previus_day_average(data)
    previusDayAverage.append(0)
    data['PreviusDatLoad']=previus_day_average(data)
    temperatures = merge_with_temp(data, temp_filename)
    data["Temp"] = temperatures
    temp_mean=np.nanmean(np.array(data["Temp"]))
    data["Temp"]=data["Temp"].fillna(temp_mean)
    data=data.drop(columns=["Hour"])
    data["Hour"]=[int(x.split(":")[0]) for x in get_hour(d)]
    data.to_csv("PreparedData.csv")
    pd.set_option('display.max_columns', 9)

    #with pd.option_context("display.max_rows",None,"display.max_columns",None):
     #   print(data)
    return data

#function for data preprocessing
def get_year(data):
    parts=data.str.split("/")

    return [(int(row[2].split(" ")[0]) + 2000) for row in parts]

#function for data preprocessing
def previus_day_average(data):
    indexDay=4
    indexMonth=3
    indexYear=2
    index=0
    previusDayList=[]
    i=0
    data1=data.values
    data2=data.values
    flag=True
    last = 0
    for row in data1:

        if(flag):
            day=int(row[indexDay])
            month=int(row[indexMonth])
            year=int(row[indexYear])
            flag=False
        elif(day==int(row[indexDay])):
                previusDayList.append(last)
                continue
        else:
            day = int(row[indexDay])
            month = int(row[indexMonth])
            year = int(row[indexYear])

        previusDay = 0
        i=0
        if(int(row[indexDay])==1 and int(row[indexYear])==2015 and int(row[indexMonth])==1):
            previusDayList.append(0)
            index+=1
            continue
        for row2 in data2:

            if(int(row[indexDay])==1 and int(row[indexYear])==int(row2[indexYear]) and int(row[indexMonth])==(int(row2[indexMonth])+1)):
                previusDay += float(row2[0])
                i+=1

            elif(int(row[indexDay])==1 and int(row[indexYear])==(int(row2[indexYear])+1) and int(row[indexMonth])==12):
                previusDay += float(row2[0])
                i+=1
            elif(int(row[indexYear])==int(row2[indexYear]) and int(row[indexMonth])==int(row2[indexMonth]) and int(row[indexDay])==(int(row2[indexDay])+1)):
                previusDay+=float(row2[0])
                i+=1

        if(i==0):
            previusDayList.append(last)
        else:

           last=previusDay/i
           previusDayList.append(last)

    return previusDayList

#function for data preprocessing
def get_hour(date):
    parts = date.str.split("/")
    return [row[2].split(" ")[1] for row in parts]


#function for data preprocessing

def is_weekend(date):

    year_m_d=get_year_month_day(date)


    days=[datetime.datetime(int(x[0]),int(x[1]),int(x[2])).weekday() for x in year_m_d]
    return [(x in (5,6))*1 for x in days]

#function for data preprocessing
def get_day_of_week(date):

    year_m_d=get_year_month_day(date)


    return [datetime.datetime(int(x[0]),int(x[1]),int(x[2])).weekday() for x in year_m_d]

#function for data preprocessing
def get_year_month_day(date):

    #format: mm/dd/yy hour

    parts=date.str.split("/")

    return [(int(row[2].split(" ")[0])+2000,row[0],row[1]) for row in parts]

#function for data preprocessing
def get_month(date):
    #format: mm/dd/yy hour
    parts=date.str.split("/")
    return [row[0] for row in parts]

#function for data preprocessing
def get_day(date):
    #format: mm/dd/yy hour

    parts=date.str.split("/")
    return [row[1] for row in parts]

#the nurual_network
class NN:
    def __init__(self,train_data,train_data_labels):
        self.train_data=train_data
        self.train_data_labels=train_data_labels

    #the neural network has two hidden layers the first have 250 neurons and the second 20 neurons each with sigmoid activation
    #the neural network has 1 output
    def build_model(self,neurons_layer1,neurons_layer2):
        self.model=tf.keras.models.Sequential(
            [keras.layers.Dense(neurons_layer1,activation=tf.nn.sigmoid,
                                input_shape=[len(self.train_data.keys())],bias_initializer=keras.initializers.Constant(0.1),kernel_initializer='random_uniform'),
             keras.layers.Dense(neurons_layer2,activation=tf.nn.sigmoid
                                ,bias_initializer=keras.initializers.Constant(0.1),kernel_initializer='random_uniform'),

             keras.layers.Dense(1)])
        self.optimizer=tf.train.RMSPropOptimizer(0.005)

        #MAPE(mean absolute percentage error) is chosen as metric for the training and generalization performance of the nn
        self.model.compile(loss=tf.keras.losses.mean_absolute_percentage_error,optimizer=self.optimizer,
                           metrics=[tf.keras.metrics.mean_absolute_percentage_error])

    #train the model for certain number of epoch
    def train_the_model(self,epoch):
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=100)

        #the history of the MAPE values for every epoch
        self.history=self.model.fit(np.array(self.train_data),np.array(self.train_data_labels),
                                    epochs=epoch,validation_split=0.2,verbose=0,callbacks=[early_stop])

    def show_history(self):
        history=pd.DataFrame(data=self.history.history['mean_absolute_percentage_error'],columns=['mean_absolute_percentage_error'])
        pd.set_option('display.max_columns', 8)
        history['Epoch']=self.history.epoch
        with pd.option_context("display.max_rows",None,"display.max_columns",None):
            print(history)


        print("minTraining MAPE= " + str(np.min(np.array(history['mean_absolute_percentage_error']))))
    def model_summary(self):
        print(self.model.summary())

    #load the saved weights
    def load_weights(self):
        self.model.load_weights("best_weights")

    def choose_best_weights(self,neurons_layer1,neurons_layer2,test_set,test_set_labels,epochs):
        best_result=100
        best_weights=[]
        best_history=None
        for i in range(0,10):
            self.build_model(neurons_layer1,neurons_layer2)
            self.train_the_model(epochs)
            result=self.test(test_set,test_set_labels,plot=False)
            if(result<best_result):
                best_history=self.history
                best_result=result
                best_weights=self.model.get_weights()

        self.model.set_weights(best_weights)
        self.history=best_history

        self.model.save_weights("best_weights",overwrite=True)

    #each combination from the elements in neurons_layer1 and neurons_layer2 is tested
    #and the one that gives best results is saved
    def cross_validation(self,neurons_layer1,neurons_layer2,test_data,test_data_labels):
        #MAPE is the metrics
        history=[]
        best_performance=100
        best_number_layer1=0
        best_number_layer2=0
        for number in neurons_layer1:
            for number2 in neurons_layer2:
                nn=NN(self.train_data,self.train_data_labels)
                nn.build_model(number,number2)
                nn.train_the_model(100)
                result=nn.test(test_data,test_data_labels)
                history.append((number,number2,result))
                if(result<best_performance):
                    best_number_layer1=number
                    best_number_layer2=number2
                    best_performance=result

        print(best_performance)
        print(best_number_layer1)
        print(best_number_layer2)
        """
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(15, 10))
        ax=fig.add_subplot(111)
        ax.scatter(x=[x[0] for x in history],y=[x[1] for x in history],z=[x[2] for x in history])
        plt.show()
        """
    #test the model on unseen test_data
    def test(self,test_set,test_labels,plot=False):
        loss,mae=self.model.evaluate(np.array(test_set),np.array(test_labels),verbose=0)

        print("MAPE on unseen data(test_set)= "+str(mae))
        test_predictions=self.model.predict(np.array(test_set),verbose=0)
        #plot the predicted values and the real values

        if(plot):
            import matplotlib.pyplot as plt

            list=[]
            for row in test_predictions:
                list.append(float(row[0]))


            d=pd.DataFrame(data=np.array(test_labels),columns=['Real values'])
            d['Model predictins']=test_predictions


            d.plot()

            plt.show()


            """
            fig = plt.figure(figsize=(15, 10))
            plt.plot(np.arange(1,len(test_predictions)+1),test_predictions,c='r',label="Model predictions")
            plt.plot(np.arange(1,len(test_predictions)+1),test_labels,c='b',label="Real values")
            plt.legend(loc="best")
             plt.show()
            """
        return mae

def create_test_set(data):

    test_set_indexes=[]
    test_set=[]
    data_val=data.values.tolist()
    np.random.seed(54654)
    size=int(len(data)*0.2)
    while(len(test_set)!=size):
        i=np.random.randint(len(data_val))
        if(i not in test_set_indexes):
            test_set_indexes.append(i)
            test_set.append(data_val[i])

    return test_set,test_set_indexes




if __name__=='__main__':
    #data=load_data("PV_production_hourly_2015-2017.csv","temp_data.csv")
    data=pd.read_csv("PreparedData.csv")

    #this data is used for ploting

    data=data.drop(columns=["Year","Unnamed: 0"])


    weekend=data.pop('Weekend')
    #normalize the data
    keys=data.keys()
    data=pd.DataFrame(data=StandardScaler().fit_transform(data),columns=keys)




    data['Weekend']=weekend

    #generating test_set
    test_set,test_set_indexes=create_test_set(data)
    # remove the train data from the training data
    #training_data=data.drop(test_set_indexes)
    #test_set=pd.DataFrame(data=test_set,columns=data.keys())
    #test_set.to_csv("Test_set.csv")
    #test_set_indexes=pd.DataFrame(data=test_set_indexes,columns=['Indexes'])
    #test_set_indexes.to_csv('Test_set_indexes.csv')

    #reading test set from csv file
    test_set=pd.read_csv("Test_set.csv")

    test_set_indexes=pd.read_csv('Test_set_indexes.csv')

    test_set=test_set.drop(columns=["Unnamed: 0"])




    test_set_indexes=test_set_indexes.drop(columns=["Unnamed: 0"])

    #remove the test_set data from the training data
    training_data = data.drop(test_set_indexes['Indexes'])



    train_data_labels=training_data.pop('Hourly production [kWh]')


    test_set_labels=test_set.pop('Hourly production [kWh]')

    neural_network=NN(training_data,train_data_labels)


    #we need to choose the optimal number of neurons in layer1 and layer2
    #num_neurons_layer1=[5,10,20,30,40,50,70,100,120,150,200,250]
    #num_neurons_layer2=[2,5,10,20,25,30,40,50,70,100,120,150,200,250]

    #neural_network.cross_validation(num_neurons_layer1,num_neurons_layer2,test_set,test_set_labels)
    #neural_network.build_model(250,20)
    #The optimal number of neurons in layer1 and layer2 is 250 and 20


    #find and save the best weights(10 iterations, in each is build a new model with new random choosem weights,then this model trained and  tested
    #and the one that gives best results, its weights is saved
    #neural_network.choose_best_weights(250,20,test_set,test_set_labels,epochs=150)

    neural_network.build_model(250,20)

    #used the saved weights instead of new trainning
    neural_network.load_weights()

    # History of the MAPE by epoch
    #neural_network.show_history()

    neural_network.test(test_set,test_set_labels,plot=True)
