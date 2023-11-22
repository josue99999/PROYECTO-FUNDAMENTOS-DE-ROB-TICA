import rbdl
import numpy as np


# Lectura del modelo del robot a partir de URDF (parsing)
modelo = rbdl.loadModel('/home/josue/PROYECTO/src/kuka_lbr_iiwa_support/urdf/modelo_dinamico.urdf')
# Grados de libertad
ndof = modelo.q_size



# Configuracion articular
q = np.array([0.5, 0.2, 0.3, 0.8, 0.5, 0.6,0.7,0.1])
# Velocidad articular
dq = np.array([0.8, 0.7, 0.8, 0.6, 0.9, 1.0,0.1,0.1])
# Aceleracion articular
ddq = np.array([0.2, 0.5, 0.4, 0.3, 1.0, 0.5,0.2,0.1])

# Arrays numpy
zeros = np.zeros(ndof)          # Vector de ceros
tau   = np.zeros(ndof)          # Para torque
g     = np.zeros(ndof)          # Para la gravedad
c     = np.zeros(ndof)          # Para el vector de Coriolis+centrifuga
M     = np.zeros([ndof, ndof])  # Para la matriz de inercia
e     = np.eye(8)               # Vector identidad

# Torque dada la configuracion del robot
rbdl.InverseDynamics(modelo, q, dq, ddq, tau)

# Parte 1: Calcular vector de gravedad, vector de Coriolis/centrifuga,
# y matriz M usando solamente InverseDynamics

# Vector gravedad: g= ID(q,0,0)
rbdl.InverseDynamics(modelo, q, zeros, zeros, g)

print('\n Vector de gravedad')
print(np.round(g, 3))

# Vector coriolis: c= (ID(q,dq,0)-g)/dq
rbdl.InverseDynamics(modelo, q, dq, zeros, c)
coriolis = c-g

print('\n Vector de coriolis')
print(np.round(c, 3))

# Matriz de inercia: M[1,:] = (ID(dq,0,e[1,:]) )/e[1,:]
for i in range(ndof):
    rbdl.InverseDynamics(modelo, q, zeros, e[i, :], M[i, :])
    M[i, :] = M[i, :]-g


print('Matriz de inercia')
print(str(np.round(M, 3)))

# Parte 2: Calcular M y los efectos no lineales b usando las funciones
# CompositeRigidBodyAlgorithm y NonlinearEffects. Almacenar los resultados
# en los arreglos llamados M2 y b2
b2 = np.zeros(ndof)          # Para efectos no lineales
M2 = np.zeros([ndof, ndof])  # Para matriz de inercia

rbdl.CompositeRigidBodyAlgorithm(modelo, q, M2)
rbdl.NonlinearEffects(modelo, q, dq, b2)


# Parte 2: Verificacion de valores
round = np.round
print("Inverse", round(M, 1))
print("Comprobar", round(M2, 1))

print("Coreolis")
print("Inverse", round(b2, 1))
print("Comprobar", round(c+g, 1))

# Parte 3: Verificacion de la expresion de la dinamica

t = M @ ddq+c+g
print("T1")
print(round(t, 1))
t2 = M2 @ ddq+b2
print("T2")
print(round(t2, 1))
print("T3")
print(round(tau, 1))
