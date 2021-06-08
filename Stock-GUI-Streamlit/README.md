This app has been deployed on streamlit. To view the app check the link below

https://share.streamlit.io/deepthisudharsan/stock-prediction-using-deep-learning/main/Stock-GUI-Streamlit/stock_gui.py

## Streamlit web app implementation of the project. 

## Pre-requisites :

Make sure to install streamlit if haven't already, to install streamlit use the following command :

```
pip install streamlit
```
All the package requirements along with the versions have been mentioned in the requirements.txt file. 

## How to run?

To run the app, in the anaconda prompt, go to the location where the stock_gui.py file is using the cd command and then run the following line:

```
streamlit run stock_gui.py
```

## Excerpts from the app

On execution, a locally hosted page pops up in the browser

![image](https://user-images.githubusercontent.com/59824729/120770616-f4754300-c53b-11eb-98ac-ac9de4f3fd91.png)

If we select the first option that is to upload the data, we get something like this. Whereas if we choose to connect to the web and get the data, we get options to select the company and range of data. 

![image](https://user-images.githubusercontent.com/59824729/120770695-0c4cc700-c53c-11eb-9495-e82ba5243b11.png)

Suppose we choose to browse and upload the data, we browse for our data file in our file uploader (which has been limited to 200MB data files). 

![image](https://user-images.githubusercontent.com/59824729/120770723-153d9880-c53c-11eb-86ab-f32abfd76fdb.png)

Here we can see that Google’s dataset has been chosen from the system.

![image](https://user-images.githubusercontent.com/59824729/120770786-225a8780-c53c-11eb-8c67-ad55977a38c2.png)

The head of the data frame is printed along with the options for choosing the period for which we want to train the model to predict a particular day’s price and also a dropdown box for selecting the Deep learning model that we want to train.

![image](https://user-images.githubusercontent.com/59824729/120770817-2d151c80-c53c-11eb-8173-abdb322fd4ad.png)

The user can choose from the five Deep Learning algorithms – LSTM, MLP, RNN, Basic ANN, and Autoencoder. Suppose the user chooses MLP, the R2 score, Mean Squared Log error, and the output plot with the predicted and observed lines are given. 

![image](https://user-images.githubusercontent.com/59824729/120770845-369e8480-c53c-11eb-83e7-f30d1c40cfbe.png)

## References

1. https://www.youtube.com/watch?v=0pHJOzNDdOo&feature=youtu.be

## Additional Note

The hyperparameters that we have used in our deep learning models show the best results for the Google Stock Price dataset that has been uploaded in the home page. The code works for any csv dataset but getting the best accuracy isn't assured. 
