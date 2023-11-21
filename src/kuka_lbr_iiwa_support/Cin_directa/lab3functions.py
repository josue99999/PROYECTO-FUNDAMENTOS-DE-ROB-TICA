import numpy as np
from copy import copy

cos=np.cos; sin=np.sin; pi=np.pi


def dh(d, theta, a, alpha):
    """
    Calcular la matriz de transformacion homogenea asociada con los parametros
    de Denavit-Hartenberg.
    Los valores d, theta, a, alpha son escalares.
    """
    # Escriba aqui la matriz de transformacion homogenea en funcion de los valores de d, theta, a, alpha
    T = np.array([[cos(theta), -1*cos(alpha)*sin(theta), sin(alpha)*sin(theta), a*cos(theta)],
                  [sin(theta), cos(alpha)*cos(theta), -1*sin(alpha)*cos(theta), a*sin(theta)],
                  [0, sin(alpha), cos(alpha), d],
                  [0,0,0,1]])
    return T


def fkine_ur10(q):
    """
    Calcular la cinematica directa del robot UR5 dados sus valores articulares. 
    q es un vector numpy de la forma [q1, q2, q3, q4, q5, q6]
    """
    # Longitudes (en metros)
    # T1 = dh(0.128, q[0], 0,pi/2) 
    # Matrices DH (completar), emplear la funcion dh con los parametros DH para cada articulacion
    T1 = dh(0.36, pi+q[0], 0, pi/2)
    T2 = dh(0.0, pi+q[1], 0, pi/2)
    T3 = dh(0.42, q[2], 0.0, pi/2)
    T4 = dh(0.0, pi+q[3], 0.0, pi/2)
    T5 = dh(0.4, q[4], 0, pi/2)
    T6 = dh(0.0, pi+q[5], 0, pi/2) 
    T7 = dh(q[6],0.0, 0, 0)
    T8 = dh(0.1, q[7],0, pi/2)
    # Efector final con respecto a la base
    #T = ((((((T1@T2)@T3)@T4)@T5)@T6)@T7)@T8
    T = ((((((T1@T2)@T3)@T4)@T5)@T6)@T7)@T8
    return T


