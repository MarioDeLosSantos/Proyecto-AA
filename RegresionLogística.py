import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
from scipy.io import loadmat

#Calculo del sigmoide 
def sigmoide(x):
    """
    Calcula el sigmoide sobre unos datos concretos.
    Args:
        x: Datos sobre los que se hara el sigmoide 
    Returns:
        sigmoide: tipo Float
    """
    return (1/(1+np.exp(-x)))

#calculo de coste de forma vectorizda
def calculaCoste(Theta,X,Y,landa):
    """
    Calcula el coste de la funcion.
    Args:
        theta: dimension del array(n+1, 1) 
        x: Conjunto de X de la funcion con dimension(m, n+1) 
        y: Conjunto de Y de la funcion con dimension(m, 1)
        landa: Parametro de reguralizacion
    Returns:
        coste: tipo Float
    """
    m=X.shape[0]
    primersumando=np.dot(np.log(sigmoide(np.dot(X,Theta))).T,Y)
    segundosumando=np.dot(np.log(1-sigmoide(np.dot(X,Theta))).T,(1-Y))
    return(-1/len(X))*(primersumando+segundosumando)+(landa/(2*m))* np.sum(np.square(Theta[:, 1:]))

#calculo del gradiente de forma vectorizada
def calculaGradiente(Theta,X,Y,landa):
    """
    Calcula el gradiente de la funcion.

    m=numero de filas de los datos de entrenameinto
    n=numero de columnas de los datos

    Args:
        theta: dimension del array(n+1, 1) 
        x: Conjunto de X de la funcion con dimension(m, n+1) 
        y: Conjunto de Y de la funcion con dimension(m, 1)
        landa: Parametro de reguralizacion
    Returns:
        gradiente: tipo array de numpy
    """
    m=X.shape[0]
    aux=np.r_[[0],Theta[1:]]
    return (1/m)*(np.dot(X.T,sigmoide(np.dot(X,Theta))-np.ravel(Y)))+(landa*aux/m)

#Devuelve tanto el coste como el gradiente para que asi podamos usar la funcion
#minimize de scipy de forma generica
def CosteyGradiente(Theta,X,Y,landa):
    """
    Calcula el gradiente de la funcion.

    m=numero de filas de los datos de entrenameinto
    n=numero de columnas de los datos

    Args:
        theta: dimension del array(n+1, 1) 
        x: Conjunto de X de la funcion con dimension(m, n+1) 
        y: Conjunto de Y de la funcion con dimension(m, 1)
        landa: Parametro de reguralizacion
    Returns:
        coste y gradiente: coste tipo float y gradiente tipo array de numpy
    """
    return (calculaCoste(Theta,X,Y,landa),calculaGradiente(Theta,X,Y,landa))

# Calculo del landa optimo 
def encontrarMejorLanda(X, y, Xval, yval):
    """
    Generamos los errores de validacion y entrenamiento con una serie de landas
    diferentes y nos quedareos con el landa que haga minima el coste en el 
    conjunto de datos de validacion

    m=numero de filas de los datos de entrenameinto
    n=numero de columnas de los datos
    s=numero de filas de los datos de validacion
    
    Args:
        X: Conjunto X de datos de entrenamiento de la funcion con dimension (m, n+1)
        y: Conjunto Y de datos de entrenamiento de la funcion con dimension (m, 1)
        Xval: Conjunto X de datos de validacion de la funcion con dimension (s, m+1)
        yval: Conjunto Y de datos de validacion de la funcion con dimension (s, 1)
    Returns:
        lambda: tipo Float
    """
    # Creamos una lista con 10 tipos de landas diferentes.
    landas = np.array([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10])

    # Inicializamos los errores.
    error_train = np.zeros((len(landas), 1))
    error_val = np.zeros((len(landas), 1))
    
    # Recorremos el vector de landas
    for i in range(len(landas)):
        landa = landas[i]

        # Entrenar el modelo con cada uno de los landas del vector
        theta = ConseguirThetaMinimo(X, y, landa,'TNC')

        # Coger cada uno de los costes con el theta encontrado y landa = 0
        error_train[i] = calculaCoste(theta,X,y,0)[0]
        error_val[i] = calculaCoste(theta,Xval,yval,0)[0]
         
    return np.amin(error_val)

def ConseguirThetaMinimo(X,y,landa,method_):
    """
    Entrena con regresion logistica los datos (X,Y)
    con el parametro de reguralizacion y con la
    tecnica algoritmica (method_),
    Devuelve el theta optimo

    m=numero de filas de los datos de entrenameinto
    n=numero de columnas de los datos

    Args:
        X: Conjunto X de datos de entrenamiento de la funcion con dimension (m, n+1)
        y: Conjunto Y de datos de entrenamiento de la funcion con dimension (m, 1)
        landa: termino de regularizacion , tipo float
        method_: tecnica algoritmica usada (en clase es TNC)
    Returns:
        theta: tipo array de numpy con dimension (n+1, )
    """

    # Inicializamos los thetas.
    initial_theta = np.zeros((X.shape[1], 1))
    
    # Usamos la funcion que devuelve tanto el coste como el gradiente para poder
    # usar la funcion de minimize generica y asi poder pasarle diferentes tipos de
    # algoritmos
    results = opt.minimize(fun=CosteyGradiente,
                       x0=initial_theta,
                       args=(X,y,landa),
                       method=method_,
                       jac=True,
                       options={'maxiter':200})
    theta = results.x
    return theta    

#Calcula un clasificador por cada etiqueta que tengamos
def oneVsAll (X, y,num_etiquetas,landa,method_):

    """
    Crea un classificador para cada tipo de etiqueta 
    tengamos que en nuestro proyecto sera para cada tipo
    de raza de perro que usemos

    m=numero de filas de los datos de entrenameinto
    n=numero de columnas de los datos
    
    Args:
        X: Conjunto X de datos de entrenamiento de la funcion con dimension (m, n+1)
        y: Conjunto Y de datos de entrenamiento de la funcion con dimension (m, 1)
        num_etiquetas: Numero de razas de perro diferentes
        landa: termino de regularizacion , tipo float
        method_: tecnica algoritmica usada (en clase es TNC)
    Returns:
        theta: tipo array de numpy con dimension (n+1, )
    """
    #lista donde guardaremos todos los clasificadores
    clasificadores=[]
   
    for i in range(num_etiquetas):
        array=(y==i+1)
        array=array.astype(int)
        clasificadores.append(array)

    #Una vez tenemos los clasificadores , hayamos el theta
    #optimo para cada uno
    thetas=[]

    for i in range(len(clasificadores)):
        thetas.append(ConseguirThetaMinimo(X,clasificadores[i],landa,method_))

    #Devolvemos tanto los clasificadores como el theta optimo 
    #para cada uno de ellos
    return thetas,clasificadores  

#Calcula el porcentaje de aciertos para cada uno de 
#los thetas optimos
def porcentajeTotal(X,thetaOptimo,Y):
    
    """
    Calcula la precision de prediccion de nuestro 
    programa calculando el porcentaje total de aciertos
    para cada uno de los thetas optimos de los clasificadores

    m=numero de filas de los datos de entrenameinto
    n=numero de columnas de los datos

    Args:
        X: Conjunto X de datos de entrenamiento de la funcion con dimension (m, n+1)
        thetaOptimo: Conjunto de thetas optimos (n+, n+1)
        y: Conjunto Y de datos de entrenamiento de la funcion con dimension (m, 1)
    Returns:
        porcentaje: tipo float
    """
    aciertos=0
    #Para cada una de las filas
    for i in range(X.shape[0]):
        thetaIesimo=0
        maxvalor=-1
        #Para cada uno de los theta optimo
        for x in range(len(thetaOptimo)):
            valor=sigmoide(np.dot(X[i],thetaOptimo[x]))
            if(valor>maxvalor):
                maxvalor=valor
                thetaIesimo=x+1
        #Comprobamos si hemos acertado o no        
        if(thetaIesimo==Y[i]):aciertos+=1     
    return(aciertos/X.shape[0])

#Comparamos el numero de aciertos obtenidos segun
#la tecnica algortimica que usemos
def ComparacionAciertos(tecnicas,X,y,numEtiquetas,landa):
    
    """
    Calcula la precision de prediccion de nuestro 
    programa calculando el porcentaje total de aciertos
    para cada uno de los thetas optimos de los clasificadores
    con todas las tecnicas algortimicas disponibles para la 
    funcion minimize de scipy

    m=numero de filas de los datos de entrenameinto
    n=numero de columnas de los datos

    Args:
        tecnicas:Conjuntos de tecnicas disponible para la funcion de minimize de scipy(TNC,CG...etc)
        X: Conjunto X de datos de entrenamiento de la funcion con dimension (m, n+1)
        y: Conjunto Y de datos de entrenamiento de la funcion con dimension (m, 1)
        numEtiquetas: Numero de razas de perro diferentes de nuestro dataset
        landa:termino de reguralizacion
       
    Returns:
        porcentajes: lista con cada uno de los porcentajes para cada una de las tecnicas
    """
    porcentajes=[]

    #Recorremos cada una de las tecnicas
    for i in range(tecnicas):
      thetasOptimos,clasificadores=oneVsAll(X,y,numEtiquetas,landa,tecnicas)
      porcentajes.append(porcentajeTotal(X,thetasOptimos,y))

    #Devolvemos la lista
    return porcentajes  

def main():
    return 0

main()
