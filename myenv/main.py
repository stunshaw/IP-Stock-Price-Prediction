import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
from keras.models import Sequential
from keras.layers import LSTM,Dropout,Dense
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('IP_stock_1yr.csv')

# see the first 5 rows
#print(df.head())
# see the last 5 rows
#print(df.tail())

# we only need the date and close/last columns
df = df[['Date', 'Close/Last']]
#print(df.head())

# remove the $ signs
df = df.replace({'\$':''}, regex = True)

# convert the Closing Price to float and Date to datetime
df = df.astype({'Close/Last': float})
df['Date'] = pd.to_datetime(df.Date, format="%m/%d/%Y")
df.dtypes

# index column
df.index = df['Date']
plt.plot(df['Close/Last'],label='Close Price')
plt.title(f"IP's Close Price history")
plt.xlabel(f'Date')
plt.ylabel(f'Price')
plt.legend()
plt.show()

# data preparation
df = df.sort_index(ascending=True,axis=0)
data = pd.DataFrame(index=range(0,len(df)),columns=['Date','Close/Last'])
for i in range(0,len(data)):
    data['Date'][i]=df['Date'][i]
    data['Close/Last'][i]=df['Close/Last'][i]
#print(data.head())

# min-max scaler
scaler=MinMaxScaler(feature_range=(0,1))
data.index=data.Date
data.drop('Date',axis=1,inplace=True)
final_data = data.values
train_data=final_data[0:200,:]
valid_data=final_data[200:,:]
scaler=MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(final_data)
x_train_data,y_train_data=[],[]
for i in range(60,len(train_data)):
    x_train_data.append(scaled_data[i-60:i,0])
    y_train_data.append(scaled_data[i,0])

# long short-term memory model
lstm_model=Sequential()
lstm_model.add(LSTM(units=50,return_sequences=True,input_shape=(np.shape(x_train_data)[1],1)))
lstm_model.add(LSTM(units=50))
lstm_model.add(Dense(1))

model_data=data[len(data)-len(valid_data)-60:].values
model_data=model_data.reshape(-1,1)
model_data=scaler.transform(model_data)

# train and test data
lstm_model.compile(loss='mean_squared_error',optimizer='adam')

x_train_data = np.array(x_train_data)
y_train_data = np.array(y_train_data)

lstm_model.fit(x_train_data,y_train_data,epochs=1,batch_size=1,verbose=2)

x_test=[]
for i in range(60,model_data.shape[0]):
    x_test.append(model_data[i-60:i,0])
x_test=np.array(x_test)
x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))

# prediction function
predicted_stock_price=lstm_model.predict(x_test)
predicted_stock_price=scaler.inverse_transform(predicted_stock_price)

# prediction result
train_data=data[:200]
valid_data=data[200:]
valid_data['Predictions']=predicted_stock_price
plt.plot(train_data['Close/Last'])
plt.plot(valid_data[['Close/Last','Predictions']])
plt.title(f"IP's Close Price Predictions")
plt.xlabel(f'Date')
plt.ylabel(f'Price')
plt.legend()
plt.show()