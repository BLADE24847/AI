from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np


input_data = np.array([[40], [30], [20], [10]], dtype=np.float32)
labels = np.array([[50], [40], [30], [20]], dtype=np.float32)  

model = Sequential()


model.add(Dense(10, input_dim=1, activation='relu'))  
model.add(Dense(1))  


model.compile(loss='mean_squared_error', optimizer='adam')


history = model.fit(input_data, labels, epochs=5500, verbose=1)


predictions = model.predict(np.array([[50]], dtype=np.float32))
print("Next number in sequence:")
print(predictions)
