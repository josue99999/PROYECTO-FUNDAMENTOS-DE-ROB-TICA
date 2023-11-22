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
    
    

def fkine_ur5(q):
    """
    Calcular la cinematica directa del robot UR5 dados sus valores articulares. 
    q es un vector numpy de la forma [q1, q2, q3, q4, q5, q6]
    """
    # Longitudes (en metros)

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



def jacobian_ur5(q, delta=0.0001):
    """
    Jacobiano analitico para la posicion. Retorna una matriz de 3x6 y toma como
    entrada el vector de configuracion articular q=[q1, q2, q3, q4, q5, q6]
    """
    # Crear una matriz 3x8
    J = np.zeros((3,8))

    # Transformacion homogenea inicial (usando q)
    T = fkine_ur5(q)
    
    # Iteracion para la derivada de cada columna
    for i in range(8):
        # Copiar la configuracion articular inicial
        dq = copy(q)
        
        # Incrementar la articulacion i-esima usando un delta
        dq[i] = dq[i] + delta

        if dq[6] >= 0.08:
            dq[6] = 0.08
        elif dq[6] <= 0.0:
            dq[6] = 0.0

        
        # Transformacion homogenea luego del incremento (q+delta)
        dT = fkine_ur5(dq)

        # Aproximacion del Jacobiano de posicion usando diferencias finitas
        J[0,i] = (dT[0,3] - T[0,3])/delta #derivadas de x
        J[1,i] = (dT[1,3] - T[1,3])/delta #derivadas de y
        J[2,i] = (dT[2,3] - T[2,3])/delta #derivadas de z
    return J


def ikine_ur5(xdes, q0):
    """
    Calcular la cinematica inversa de UR5 numericamente a partir de la configuracion articular inicial de q0. 
    Emplear el metodo de newton
    """
    epsilon  = 0.001
    max_iter = 1000
    delta    = 0.00001
    E = np.zeros((3,max_iter))

    q  = copy(q0)
    for i in range(max_iter):
        #main loop
        T = fkine_ur5(q)
        x = T[0:3,3]

        # error
        e = xdes - x
#        E[i,:] = e

        #Calculo del nuevo q:
        J = jacobian_ur5(q, delta)
        Jinv = np.linalg.pinv(J)
        
        q = q + Jinv@e

        if q[6] >= 0.08:
            q[6] = 0.08
        elif q[6] <= 0.0:
            q[6] = 0.0

        
        #Condicion de cierre
        if (np.linalg.norm(e) < epsilon):
            print("\nValores articulares obtenidos: ", np.round(q,4))
            print("\nNumero de iteraciones: ", i)
            break

        if (i == max_iter-1):
            print("\nNo se encontro solucion en ", i, "iteraciones.")
            q = q-q
    return q

def ik_gradient_ur5(xdes, q0, alfa =0.05):
    """
    Calcular la cinematica inversa de UR5 numericamente a partir de la configuracion articular inicial de q0. 
    Emplear el metodo gradiente
    """
    epsilon  = 0.001
    max_iter = 100000
    delta    = 0.00001
    
    E = np.zeros((3,max_iter))

    q  = copy(q0)
    for i in range(max_iter):
        #main loop
        T = fkine_ur5(q)
        x = T[0:3,3]

        # error
        e = xdes - x
        #E[i,:] = e

        #Calculo del nuevo q:
        J = jacobian_ur5(q, delta)
        
        q = q + alfa*(J.T@e)

        if q[6] >= 0.08:
            q[6] = 0.08
        elif q[6] <= 0.0:
            q[6] = 0.0

        
        #Condicion de cierre
        if (np.linalg.norm(e) < epsilon):
            print("\nValores articulares obtenidos: ", np.round(q,4))
            print("\nNumero de iteraciones: ", i)
            break

        if (i == max_iter-1):
            print("\nNo se encontro solucion en ", i, "iteraciones.")
            q = q-q
        
    return q
