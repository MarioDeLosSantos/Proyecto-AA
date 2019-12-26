import numpy as np
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy.io import loadmat
from checkNNGradients import checkNNGradients

counter = 0
# Cáculo del coste no regularizado
def CalculoCoste(m, h, y):
    """
    Calcula el coste sin regularizar de la funcion
    Args:
        m: numero de filas de los datos de entrenameinto 
        h: Es la matriz resultado de aplicar el el sigmoide a z3 que es la matriz de pesos de la segunda capa 
        y: Conjunto Y de datos de entrenamiento de la funcion con dimension (m, 1)
    Returns:
        Coste sin regularizar: tipo float 
    """
    J = 0
    for i in range(m):
        J += np.sum(-y[i] * np.log(h[i]) \
             - (1 - y[i]) * np.log(1 - h[i]))
    return (J / m)


# Cálculo del coste regularizado
def CalculoCosteRegularizado(m, h, Y, reg, theta1, theta2):
    """
    Calcula el coste regularizado de la funcion usando para ello el sin regularizar
    Args:
        m: numero de filas de los datos de entrenameinto 
        h: Es la matriz resultado de aplicar el el sigmoide a z3 que es la matriz de pesos de la segunda capa 
        y: Conjunto Y de datos de entrenamiento de la funcion con dimension (m, 1)
        reg: factor de regularizacion
        theta1: matriz de thetas de la primera capa
        theta2: matriz de thetas de la segunda capa
    Returns:
        Coste regularizado: tipo float 
    """
    return (CalculoCoste(m, h, Y) + 
        ((reg / (2 * m)) * 
        (np.sum(np.square(theta1[:, 1:])) + 
        np.sum(np.square(theta2[:, 1:])))))

# Función sigmoide
def sigmoid(z):
    """
    Calcula el sigmoide sobre unos datos concretos.
    Args:
        x: Datos sobre los que se hara el sigmoide 
    Returns:
        sigmoide: tipo Float
    """
    return 1 / (1 + np.exp(-z))

# Inicializa una matriz de pesos aleatorios
def pesosAleatorios(L_in, L_out):
    """
    Inicializa la matriz de theta de dimensiones
    recibidas por parametro con valores aleatorios 
    Args:
        L_in: tamaño en columnas de la matriz de thetas
        L_out: tamaño en filas de la matriz de thetas 
    Returns:
        theta: Matriz de thetas con pesos aleatorios 
    """
    ini = 0.12
    theta = np.random.uniform(low=-ini, high=ini, size=(L_out, L_in))
    theta = np.hstack((np.ones((theta.shape[0], 1)), theta))
    return theta

# Devuelve "Y" a partir de una X y no unos pesos determinados
def PropagacionHaciaDelante(X, theta1, theta2):
    """
    calcula los pesos de las distintas capas ocultas y 
    devuelve los datos resultantes
    Args:
        X:C
        theta1: Matriz de thetas para la primera capa
        theta2: Matriz de thetas para la segunda capa 
    Returns:
        a1: Matriz X con la primera columna de unos
        z2: Matriz de pesos calculada sobre la capa dos
        a2: Matriz resultante de aplicar sigmoide a z1 a la que le añadimos la primera columna de unos
        z3: Matriz de pesos calculada sobre la capa tres
        h: Es la matriz resultado de aplicar el el sigmoide a z3
    """

    m = X.shape[0] 

    a1 = np.hstack([np.ones([m, 1]), X])    # (90000, 797)
    z2 = np.dot(a1, theta1.T)   # (5000, 30)

    a2 = np.hstack([np.ones([m, 1]), sigmoid(z2)])  # (90000, 31)
    z3 = np.dot(a2, theta2.T)   # (90000, 10)

    h = sigmoid(z3) # (90000, 10)

    return a1, z2, a2, z3, h

    # Devuelve el coste y el gradiente de una red neuronal de dos capas
def backprop(params_rn, num_entradas, num_ocultas, num_etiquetas, X, y, reg):    
    m = X.shape[0]
    print("working")
    # Sacamos ambas thetas
    theta1 = np.reshape(params_rn[:num_ocultas * (num_entradas + 1)],
            (num_ocultas, (num_entradas + 1)))

    theta2 = np.reshape(params_rn[num_ocultas * (num_entradas + 1): ], 
        (num_etiquetas, (num_ocultas + 1)))

    a1, z2, a2, z3, h = PropagacionHaciaDelante(X, theta1, theta2)  

    coste = CalculoCosteRegularizado(m, h, y, reg, theta1, theta2) # Coste regularizado

    # Inicialización de dos matrices "delta" a 0 con el tamaño de los thethas respectivos
    delta1 = np.zeros_like(theta1)
    delta2 = np.zeros_like(theta2)

    # Por cada ejemplo
    for t in range(m):
        a1t = a1[t, :] 
        a2t = a2[t, :] 
        ht = h[t, :] 
        yt = y[t]

        d3t = ht - yt
        d2t = np.dot(theta2.T, d3t) * (a2t * (1 - a2t)) 

        delta1 = delta1 + np.dot(d2t[1:, np.newaxis], a1t[np.newaxis, :])
        delta2 = delta2 + np.dot(d3t[:, np.newaxis], a2t[np.newaxis, :])

    delta1 = delta1 / m
    delta2 = delta2 / m

    # Gradiente perteneciente a cada delta
    delta1[:, 1:] = delta1[:, 1:] + (reg * theta1[:, 1:]) / m
    delta2[:, 1:] = delta2[:, 1:] + (reg * theta2[:, 1:]) / m
    
    #Unimos los gradientes
    gradiente = np.concatenate((np.ravel(delta1), np.ravel(delta2)))

    return coste, gradiente

#calculo del porcentaje
def porcentajeTotal(h,y):
    """
    Calcula la precision de prediccion de nuestro 
    programa calculando el porcentaje total de aciertos
    Args:

    h: Es la matriz resultado de aplicar el el sigmoide a z3
    y: Conjunto Y de datos de entrenamiento de la funcion con dimension (m, 1)
    Returns:
        porcentaje: tipo float
    """
    aciertos = 0
    for i in range (h.shape[0]):
        max = np.argmax(h[i])
        if max == y[i]:
            aciertos += 1
    precision = (aciertos / h.shape[0]) * 100        
    return (precision)  

def main():
    #Cargamos el dataset
    data = loadmat("data20x20.mat")

    #guardamos la matriz y en un solo vector 
    y = data["y"].ravel()
    X = data["X"]

    #el numero de entradas son cada uno de los píxeles de la imagen de 300x300
    num_entradas = X.shape[1]
    num_ocultas = 150
    num_etiquetas = 10

    # Transforma Y en una matriz de vectores, donde cada vector está formado por todo 
    # 0s excepto el valor marcado en Y, que se pone a 1
    # 3 ---> [0, 0, 0, 1, 0, 0, 0, 0, 0, 0] La matriz tiene tantas filas como componentes tiene y
    # 5 ---> [0, 0, 0, 0, 0, 1, 0, 0, 0, 0] estando a uno la posicion y[i]
    lenY = len(y)
    y = (y - 1)
    y_onehot = np.zeros((lenY, num_etiquetas))
    for i in range(lenY):
        y_onehot[i][y[i]] = 1
    

    # Inicialización de dos matrices de pesos de manera aleatoria
    Theta1 = pesosAleatorios(num_entradas, num_ocultas) # (25, 401)
    Theta2 = pesosAleatorios(num_ocultas, num_etiquetas) # (10, 26)

    # Crea una lista de Thetas
    Thetas = [Theta1, Theta2]

    # Concatenación de las matrices de pesos en un solo vector
    unrolled_Thetas = [Thetas[i].ravel() for i,_ in enumerate(Thetas)]
    nn_params = np.concatenate(unrolled_Thetas)

    # Obtención de los pesos óptimos entrenando una red con los pesos aleatorios
    optTheta = opt.minimize(fun=backprop, x0=nn_params, 
            args=(num_entradas, num_ocultas, num_etiquetas,
            X, y_onehot, 1), method='TNC', jac=True,
            options={'maxiter': 500})

    # Desglose de los pesos óptimos en dos matrices
    Theta1Final = np.reshape(optTheta.x[:num_ocultas * (num_entradas + 1)],
        (num_ocultas, (num_entradas + 1)))

    Theta2Final = np.reshape(optTheta.x[num_ocultas * (num_entradas + 1): ], 
        (num_etiquetas, (num_ocultas + 1)))

    # H, resultado de la red al usar los pesos óptimos
    a1, z2, a2, z3, h = PropagacionHaciaDelante(X, Theta1Final, Theta2Final) 
    
    # Cálculo de la precisión de la red neuronal
    print("{0:.2f}% de precision".format(porcentajeTotal(h,y)))

main()
