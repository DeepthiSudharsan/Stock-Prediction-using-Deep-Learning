# Importing libraries
import streamlit as st
import datetime
from pandas_datareader import data as pdr
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM,Dropout
from sklearn.metrics import accuracy_score
from tensorflow.keras import datasets, layers,models
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Activation,SimpleRNN
from sklearn.neural_network import MLPRegressor
from sklearn import metrics
tf.random.set_seed(7)

st.title('Stock Prediction using Deep Learning')
st.subheader('Select the method of input:')
# Adding radio buttons for the user to choose between Uploading csv and getting stock data from the net 
option = st.radio('Radio', ["Upload the data (.csv format)","Get data from the net"])
# creating a side bar 
st.sidebar.title("Created By:")
st.sidebar.subheader("Deepthi Sudharsan")
st.sidebar.subheader("Meghna B Menon")
# Adding an image from Unsplash to the side bar 
st.sidebar.image("https://bit.ly/2RgH8BC", width=None)
st.sidebar.markdown("Photo by Carlos Muza on Unsplash")
# class for the deep learning models
class stock_predict_DL:
    
    def __init__(self,comp_df):
        # reseved method in python classes (Constructor)
        # We are taking only the Open prices for predicting 
        data = comp_df.filter(['Open'])
        dataset = data.values
        # We take 90% of the data for training and 10% for testing 
        st.subheader('How much percent of the data needs to be allocated for training?')
        st.text('Default is set to 90')
        perc_train = st.number_input('',step = 1,min_value=1, value = 90)
        training_data_len = int(np.ceil( len(dataset) * (perc_train/100)))
        # We are scaling the open prices to the range(0,1)
        self.scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = self.scaler.fit_transform(dataset)
        # Taking the first 90% of the dataset for training 
        train_data = scaled_data[0:int(training_data_len), :]
        # Split the data into self.X_train and self.y_train data sets
        self.X_train = []
        self.y_train = []
        
        # We are taking predicting the open price of a given day based on the trend in the previous 60 days
        for i in range(k, len(train_data)):
            self.X_train.append(train_data[i-k:i, 0])
            self.y_train.append(train_data[i, 0])

        # Convert the self.X_train and self.y_train to numpy arrays 
        self.X_train, self.y_train = np.array(self.X_train), np.array(self.y_train)

        # Create the testing data set
        # Create a new array containing scaled values from index 1543 to 2002 
        test_data = scaled_data[training_data_len - k: , :]
        # Create the data sets self.X_test and self.y_test
        self.X_test = []
        # Rmaining 10% of the data needs to be given for testing 
        self.y_test = dataset[training_data_len:, :]
        for i in range(k, len(test_data)):
            self.X_test.append(test_data[i-k:i, 0])

        # Convert the data to a numpy array
        self.X_test = np.array(self.X_test)
        test_dates = comp_df['Date'].values
        self.testd = test_dates[training_data_len:] # stores the test dates
        
    def LSTM_model(self):
        
        st.title("Long Short-Term Memory (LSTM)")
        # Reshape the data
        Xtrain = np.reshape(self.X_train, (self.X_train.shape[0], self.X_train.shape[1], 1))
        # Reshape the data
        Xtest = np.reshape(self.X_test, (self.X_test.shape[0], self.X_test.shape[1], 1 ))
        # Build the LSTM model
        model = Sequential()
        model.add(LSTM(128, return_sequences=True, input_shape= (Xtrain.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(32))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        # We are adding dropout to reduce overfitting 
        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')
        # Train the model
        model.fit(Xtrain, self.y_train, batch_size=1, epochs= 1)
         # Get the models predicted price values 
        predictions = model.predict(Xtest)
        # We need to inverse transform the scaled data to compare it with our unscaled y_test data
        predictions = self.scaler.inverse_transform(predictions)
        st.text("R2 SCORE")
        st.text(metrics.r2_score(self.y_test, predictions))
        st.text("MSLE")
        st.text(metrics.mean_squared_log_error(self.y_test, predictions))
        plt.plot(predictions)
        plt.plot(self.y_test)
        plt.legend(["Predicted","Observed"])
        plt.xticks(range(0,len(self.y_test),50),self.testd,rotation=45)
        plt.xlabel('Date',fontsize=18)
        plt.ylabel('Price',fontsize=18)
        plt.title("LSTM")
        st.pyplot(plt)
        
    def autoen_model(self):
        
        st.title("Autoencoder")
        # No of encoding dimensions
        encoding_dim = 32
        input_dim = self.X_train.shape[1]
        input_layer = Input(shape=(input_dim, ))
        # Encoder
        encoder = Dense(encoding_dim, activation="tanh", activity_regularizer=regularizers.l1(1e-5))(input_layer)
        encoder = Dense(int(encoding_dim / 2), activation="relu")(encoder)
        # Decoder
        decoder = Dense(int(encoding_dim / 2), activation='tanh')(encoder)
        decoder = Dense(1, activation='relu')(decoder)
        autoencoder = Model(inputs=input_layer, outputs=decoder)
        nb_epoch = 10
        b_size = 32
        # Fitting and compiling the train data using adam (stochastic gradient) optimiser and mse loss
        autoencoder.compile(optimizer='adam',loss='mean_squared_error')
        autoencoder.fit(self.X_train, self.y_train,epochs=nb_epoch,batch_size = b_size,shuffle=True)
        predictions = autoencoder.predict(self.X_test)
        predictions = self.scaler.inverse_transform(predictions)
        st.text("R2 SCORE")
        st.text(metrics.r2_score(self.y_test, predictions))
        st.text("MSLE")
        st.text(metrics.mean_squared_log_error(self.y_test, predictions))
        plt.plot(predictions)
        plt.plot(self.y_test)
        plt.legend(["Predicted","Observed"])
        plt.xticks(range(0,len(self.y_test),50),self.testd,rotation=45)
        plt.xlabel('Date',fontsize=18)
        plt.ylabel('Price',fontsize=18)
        plt.title("AUTOENCODER")
        st.pyplot(plt)
        
    def Mlp_model(self):
        
        st.title("Multilayer perceptron (MLP)")
        # We are using MLPRegressor as the problem at hand is a regression problem
        regr = MLPRegressor(hidden_layer_sizes = 100, alpha = 0.01,solver = 'lbfgs',shuffle=True)
        regr.fit(self.X_train, self.y_train)
        # predicting the price
        y_pred = regr.predict(self.X_test)
        y_pred = y_pred.reshape(len(y_pred),1)
        y_pred = self.scaler.inverse_transform(y_pred)
        st.text("R2 SCORE")
        st.text(metrics.r2_score(self.y_test, y_pred))
        st.text("MSLE")
        st.text(metrics.mean_squared_log_error(self.y_test, y_pred))
        plt.plot(y_pred)
        plt.plot(self.y_test)
        plt.legend(["Predicted","Observed"])
        plt.xticks(range(0,len(self.y_test),50),self.testd,rotation=45)
        plt.xlabel('Date',fontsize=18)
        plt.ylabel('Price',fontsize=18)
        plt.title("MLP")
        st.pyplot(plt)
        
    def basic_ann_model(self):
        
        st.title("Basic Artificial Neural Network (ANN)")
        classifier = Sequential()
        classifier.add(Dense(units = 128, activation = 'relu', input_dim = self.X_train.shape[1]))
        classifier.add(Dropout(0.2))
        classifier.add(Dense(units = 64))
        classifier.add(Dropout(0.2))
        classifier.add(Dense(units = 1))
        # We are adding dropout to reduce overfitting
        # adam is one of the best optimzier for DL as it uses stochastic gradient method
        # Mean Square Error (MSE) is the most commonly used regression loss function.
        # MSE is the sum of squared distances between our target variable and predicted values.
        classifier.compile(optimizer = 'adam', loss = 'mean_squared_error')
        classifier.fit(self.X_train, self.y_train, batch_size = 32, epochs = 10)
        # Predicting the prices
        prediction = classifier.predict(self.X_test)
        y_pred = self.scaler.inverse_transform(prediction)
        st.text("R2 SCORE")
        st.text(metrics.r2_score(self.y_test, y_pred))
        st.text("MSLE")
        st.text(metrics.mean_squared_log_error(self.y_test, y_pred))
        plt.plot(y_pred)
        plt.plot(self.y_test)
        plt.legend(["Predicted","Observed"])
        plt.xticks(range(0,len(self.y_test),50),self.testd,rotation=45)
        plt.xlabel('Date',fontsize=18)
        plt.ylabel('Price',fontsize=18)
        plt.title("ANN")
        st.pyplot(plt)
    
    def rnn_model(self):
        
        st.title("Recurrent neural network (RNN)")
        # Reshape the data
        Xtrain = np.reshape(self.X_train, (self.X_train.shape[0], self.X_train.shape[1], 1))
        # Reshape the data
        Xtest = np.reshape(self.X_test, (self.X_test.shape[0], self.X_test.shape[1], 1 ))
        model = Sequential()
        model.add(SimpleRNN(units=4, input_shape=(Xtrain.shape[1], Xtrain.shape[2])))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(Xtrain, self.y_train, epochs=10, batch_size=1)
        # predicting the opening prices
        prediction = model.predict(Xtest)
        y_pred = self.scaler.inverse_transform(prediction)
        st.text("R2 SCORE")
        st.text(metrics.r2_score(self.y_test, y_pred))
        st.text("MSLE")
        st.text(metrics.mean_squared_log_error(self.y_test, y_pred))
        plt.plot(y_pred)
        plt.plot(self.y_test)
        plt.legend(["Predicted","Observed"])
        plt.xticks(range(0,len(self.y_test),50),self.testd,rotation=45)
        plt.xlabel('Date',fontsize=18)
        plt.ylabel('Price',fontsize=18)
        plt.title("RNN")
        st.pyplot(plt)

# flag marker
flag = "False"

# if the user chooses to get data from the net
if option == "Get data from the net":

    # Sidebar
    st.sidebar.subheader('Query parameters')
    # User can choose the start date,the end date and the company's ticker whose data they want to train
    start_date = st.sidebar.date_input("Start date", datetime.date(2012, 5, 18))
    end_date = st.sidebar.date_input("End date", datetime.date(2021,3, 25))
    # Retrieving tickers data
    ticker_list = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/s-and-p-500-companies/master/data/constituents_symbols.txt')
    tickerSymbol = st.sidebar.selectbox('Stock ticker', ticker_list) # Select ticker symbol
    # getting stock data from yahoo for the selected company (based on the ticker)
    # Since the index for the data would be the dates, we are resetting the index so that we get Date as a separate part of the data
    data = pdr.get_data_yahoo(tickerSymbol, start = start_date, end = end_date).reset_index()
    # Keeping only the dates and removing the timestamp
    data['Date'] = pd.to_datetime(data['Date']).dt.date
    # flag is set to true as data has been successfully read
    flag = "True"
    st.header('**Stock data**')
    st.write(data)

# if the user chooses to upload the data
elif option == "Upload the data (.csv format)":
    file = st.file_uploader('Dataset')
    # browsing and uploading the dataset (strictly in csv format)
    if file is not None:
        
        data = pd.read_csv(file)
        # flag is set to true as data has been successfully read
        flag = "True"
        st.header('**Stock data**')
        st.write(data)

if(flag == "True"): 
    # by default it is set that the stock price of a particular day is to be predicted based on the trend of the previous 60 days
    # the user is free to input the time frame based on which they want to predict a particular day's stock
    st.subheader('Define time window length:')
    st.text('Default is set to 60')
    k = st.number_input('',step = 1,min_value=1, value = 60)
    # creating an object of the class
    company_stock = stock_predict_DL(data)
    # The user can select which deep learning model they would like to train
    # based on the user's choice, the respective function is called
    st.subheader('Which Deep Learning model would you like to train? :')
    mopt = st.selectbox('', ["Click to select", "LSTM","MLP","RNN","Basic ANN","Autoencoder"])

    if mopt=="LSTM":
        company_stock.LSTM_model()

    if mopt=="MLP":
        company_stock.Mlp_model()

    if mopt == "RNN":
        company_stock.rnn_model()

    if mopt=="Autoencoder":
        company_stock.autoen_model()

    if mopt == "Basic ANN":
        company_stock.basic_ann_model()
