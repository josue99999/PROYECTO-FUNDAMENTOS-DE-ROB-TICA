#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import JointState
from markers import *
from functions import *
from roslib import packages

import rbdl


rospy.init_node("control_pdg")
pub = rospy.Publisher('joint_states', JointState, queue_size=1000)
bmarker_actual  = BallMarker(color['RED'])
bmarker_deseado = BallMarker(color['GREEN'])
# Archivos donde se almacenara los datos
fqact = open("/tmp/qactual.txt", "w")
fqdes = open("/tmp/qdeseado.txt", "w")
fxact = open("/tmp/xactual.txt", "w")
fxdes = open("/tmp/xdeseado.txt", "w")

# Nombres de las articulaciones
jnames = ['joint_a1', 'joint_a2','joint_a3', 'joint_a4', 'joint_a5','joint_a6','joint_a7','joint_a8']
 
# Objeto (mensaje) de tipo JointState
jstate = JointState()
# Valores del mensaje
jstate.header.stamp = rospy.Time.now()
jstate.name = jnames

# =============================================================
q = np.array([0.0, -1.0, 1.7, -2.2, -1.6, 0.0, 0.1, 0.4])
# Velocidad inicial
dq = np.array([0., 0., 0., 0., 0., 0., 0., 0.])
# Aceleracion inicial
ddq = np.array([0., 0., 0., 0., 0., 0., 0., 0.])
# Configuracion articular deseada
qdes = np.array([1.0, -1.0, 1.0, 1.3, -1.5, 1.0, 0.1, 0.2])
# Velocidad articular deseada
dqdes = np.array([0., 0., 0., 0., 0., 0., 0., 0.])
# Aceleracion articular deseada
ddqdes = np.array([0., 0., 0., 0., 0., 0., 0., 0.])
# =============================================================

# Posicion resultante de la configuracion articular deseada
xdes = ur5_fkine(qdes)[0:3,3]
# Copiar la configuracion articular en el mensaje a ser publicado
jstate.position = q
pub.publish(jstate)

# Modelo RBDL
modelo = rbdl.loadModel('/home/josue/PROYECTO/src/kuka_lbr_iiwa_support/urdf/modelo_dinamico.urdf')
ndof   = modelo.q_size     # Grados de libertad
zeros = np.zeros(ndof)     # Vector de ceros

# Frecuencia del envio (en Hz)
freq = 20
dt = 1.0/freq
rate = rospy.Rate(freq)

# Simulador dinamico del robot
robot = Robot(q, dq, ndof, dt)

# Bucle de ejecucion continua
t = 0.0

# Se definen las ganancias del controlador
valores = 0.1*np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
Kp = np.diag(valores)
Kd = 2*np.sqrt(Kp)

while not rospy.is_shutdown():

    # Leer valores del simulador
    q  = robot.read_joint_positions()
    if q[6] >= 0.08:
        q[6] = 0.08
    elif q[6] <= 0.0:
        q[6] = 0.0
    dq = robot.read_joint_velocities()
    # Posicion actual del efector final
    x = ur5_fkine(q)[0:3,3]
    # Tiempo actual (necesario como indicador para ROS)
    jstate.header.stamp = rospy.Time.now()

    # Almacenamiento de datos
    fxact.write(str(t)+' '+str(x[0])+' '+str(x[1])+' '+str(x[2])+'\n')
    fxdes.write(str(t)+' '+str(xdes[0])+' '+str(xdes[1])+' '+str(xdes[2])+'\n')
    fqact.write(str(t)+' '+str(q[0])+' '+str(q[1])+' '+ str(q[2])+' '+ str(q[3])+' '+str(q[4])+' '+str(q[5])+ ' '+str(q[6])+' '+str(q[7])+ '\n ')
    fqdes.write(str(t)+' '+str(qdes[0])+' '+str(qdes[1])+' '+ str(qdes[2])+' '+ str(qdes[3])+' '+str(qdes[4])+' '+str(qdes[5])+' '+str(qdes[6])+' '+str(qdes[7])+'\n ')

    # ----------------------------
    # Control dinamico (COMPLETAR)
    # ----------------------------
    # Arrays numpy
    zeros = np.zeros(ndof)          # Vector de ceros
    tau = np.zeros(ndof)          # Para torque
    g = np.zeros(ndof)          # Para la gravedad
    c = np.zeros(ndof)          # Para el vector de Coriolis+centrifuga
    M = np.zeros([ndof, ndof])  # Para la matriz de inercia
    e = np.eye(8)               # Vector identidad


    # Torque dada la configuracion del robot
    rbdl.InverseDynamics(modelo, q, dq, ddq, tau)

    # Parte 1: Calcular vector de gravedad, vector de Coriolis/centrifuga,
    # y matriz M usando solamente InverseDynamics


    # Vector gravedad: g= ID(q,0,0)
    rbdl.InverseDynamics(modelo, q, zeros, zeros, g)


    # Vector coriolis: c= (ID(q,dq,0)-g)/dq
    rbdl.InverseDynamics(modelo, q, dq, zeros, c)
    coriolis = c-g


    # Matriz de inercia: M[1,:] = (ID(dq,0,e[1,:]) )/e[1,:]
    for i in range(ndof):
        rbdl.InverseDynamics(modelo, q, zeros, e[i, :], M[i, :])
        M[i, :] = M[i, :]-g


    e= qdes-q
    de=dqdes-dq
    dde=ddqdes-ddq

    Kde=Kd.dot(de)
    Kpe=Kp.dot(e)
    u= M.dot(ddqdes+Kde+Kpe)+c.dot(ddq)+g #ley de control

    # Simulacion del robot
    robot.send_command(u)

    # Publicacion del mensaje
    jstate.position = q
    pub.publish(jstate)
    bmarker_deseado.xyz(xdes)
    bmarker_actual.xyz(x)
    t = t+dt
    # Esperar hasta la siguiente  iteracion
    rate.sleep()

fqact.close()
fqdes.close()
fxact.close()
fxdes.close()
