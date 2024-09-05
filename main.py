import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.api.models import Sequential
from keras.api.layers import Dense
from keras.api.optimizers import SGD

data = pd.read_csv("altura_peso.csv", sep=",")

x = data['Altura'].values
y = data['Peso'].values

x = x / max(x)
y = y / max(y)

model = Sequential()
model.add(Dense(1, input_dim=1, activation='linear'))

optimizer = SGD(learning_rate=0.0004)

model.compile(optimizer=optimizer, loss='mse')

history = model.fit(x, y, epochs=10000, batch_size=len(x), verbose=0)

weights = model.get_weights()
w = weights[0][0][0]
b = weights[1][0]

plt.plot(history.history['loss'])
plt.title('Error cuadrático medio vs. Número de épocas')
plt.xlabel('Número de épocas')
plt.ylabel('Error cuadrático medio (MSE)')
plt.show()

plt.scatter(x * max(x), y * max(y), label='Datos originales', color='blue')
plt.plot(x * max(x), model.predict(x) * max(y), color='red', label=f'Recta de regresión\nw={w:.4f}, b={b:.4f}')
plt.title('Regresión Lineal: Altura vs. Peso')
plt.xlabel('Altura (cm)')
plt.ylabel('Peso (kg)')
plt.legend()
plt.show()

altura_test = 170 / max(x)
peso_predicho = model.predict(np.array([altura_test]))[0][0] * max(y)
print(f'Predicción del peso para una altura de 170 cm: {peso_predicho:.2f} kg')
