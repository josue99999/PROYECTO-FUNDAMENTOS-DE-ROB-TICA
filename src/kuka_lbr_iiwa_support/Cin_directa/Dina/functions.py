import numpy as np
from copy import copy
import rbdl

pi = np.pi


class Robot(object):
    def __init__(self, q0, dq0, ndof, dt):
        self.q = q0    # numpy array (ndof x 1)
        self.dq = dq0  # numpy array (ndof x 1)
        self.M = np.zeros([ndof, ndof])
        self.b = np.zeros(ndof)
        self.dt = dt
        self.robot = rbdl.loadModel('/home/josue/PROYECTO/src/kuka_lbr_iiwa_support/urdf/modelo_dinamico.urdf')

    def send_command(self, tau):
        rbdl.CompositeRigidBodyAlgorithm(self.robot, self.q, self.M)
        rbdl.NonlinearEffects(self.robot, self.q, self.dq, self.b)
        ddq = np.linalg.inv(self.M).dot(tau-self.b)
        self.q = self.q + self.dt*self.dq
        self.dq = self.dq + self.dt*ddq

    def read_joint_positions(self):
        return self.q

    def read_joint_velocities(self):
        return self.dq


def dh(d, theta, a, alpha):
    """
    Matriz de transformacion homogenea asociada a los parametros DH.
    Retorna una matriz 4x4
    """
    sth = np.sin(theta)
    cth = np.cos(theta)
    sa  = np.sin(alpha)
    ca  = np.cos(alpha)
    T = np.array([[cth, -ca*sth,  sa*sth, a*cth],
                   [sth,  ca*cth, -sa*cth, a*sth],
                   [0.0,      sa,      ca,     d],
                   [0.0,     0.0,     0.0,   1.0]])
    return T


def ur5_fkine(q):
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
