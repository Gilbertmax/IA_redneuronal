from keras.models import Sequential
from keras.layers import Dense
import numpy as np

#Datos de entrenamiento
x_train = np.array([[0,0],[0,1],[1,0],[1,1]])
y_train = np.array([[0],[1],[1],[0]])

#Modelo de red neuronal
model = Sequential()
model.add(Dense(4, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#Entrenar modelo
model.fit(x_train, y_train, epochs=500)

#Predecir valores
predictions = model.predict(x_train)
print(predictions)
