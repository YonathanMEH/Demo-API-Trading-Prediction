
# RED NEURONAL RECURRENTE

# Pre Procesamiento de Datos

# Importando Librerias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
class Alumnos:
	file="/static/csv/pruebas_acciones_google.csv"
	version="1.0.0"
	def __init__(self):
		pass

dataset_train = pd.read_csv('train_acciones_google.csv')
training_set = dataset_train.iloc[:, 1:2].values

# Escalado de Caracteristicas (Normalizacion)
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))
training_set_escalado = sc.fit_transform(training_set)

# Estructura con rango de 60 pasos temporales y 1 salida
X_train = []
y_train = []
for i in range(60,1258):
    X_train.append(training_set_escalado[i-60:i,0])
    y_train.append(training_set_escalado[i,0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Rediseño
X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))

# Armando Modelo RNR
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Inicializando Modelo
regresor = Sequential()

# Agregando Primera Capa LSTM y Regularizacion de Desercion
regresor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1],1)))
regresor.add(Dropout(0.2))

# Agregando 2da Capa LSTM
regresor.add(LSTM(units=50, return_sequences=True))
regresor.add(Dropout(0.2))

# Agregando 3ra Capa LSTM
regresor.add(LSTM(units=50, return_sequences=True))
regresor.add(Dropout(0.2))

# Agregando 4ta Capa LSTM
regresor.add(LSTM(units=50))
regresor.add(Dropout(0.2))

# Agregando Capa de Salida
regresor.add(Dense(units=1))

# Optimizador y Funcion de Perdida (Compilacion)
regresor.compile(optimizer='adam',loss='mean_squared_error')



# PARTE 2 - ENCAJANDO EN SET DE ENTRENAMIENTO

#Encajando Red Neuronal en Set de Entrenamiento
regresor.fit(X_train, y_train,epochs=100,batch_size=32)


# PARTE 3 - PREDICCION y VISUALIZACION

# Consiguiendo Datos Reales Precio Google
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
test_set = dataset_test.iloc[:, 1:2].values

# Prediciendo Accion Google Enero 2017
dataset_total = pd.concat((dataset_train['Open'],dataset_test['Open']),axis=0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

# Agregando Dimension Extra
X_test = []
for i in range(60,80):
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))

# Prediccion
prediccion_precio = regresor.predict(X_test)
prediccion_precio = sc.inverse_transform(prediccion_precio)


# Visualizacion
plt.plot(test_set, color='red',label='Precio Accion Real')
plt.plot(prediccion_precio, color='blue',label='Precio Accion Predecido')
plt.title('Prediccion Precio Accion Google')
plt.xlabel('Tiempo')
plt.ylabel('Precio')
plt.legend()
plt.show()































