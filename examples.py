import numpy as np
import tensorflow as tf

class CLF_10dim:
    """
    A class representing a 10-dimensional system \dot{x} = f(x,u) = vf(x) + vg(x) u  

    Methods:
        
        vf(x):
            Computes the vector field vf(x) of the system at a given state `x`.

        vg(x):
            Computes the control influence matrix vg(x) at a given state `x`.

        f(x, u):
            Computes the system dynamics f(x,u) with control input `u` at state `x`.
    """
    title = 'CLF_10dim'

    # define the dimensions of the problem
    inputdim = 10  # dimension of the state space (variable n in paper)
    interval_size = 1  # compute a Lyapunov function on the cube [-interval_size, interval_size]**inputdim
    control_size = 1  # assume the space of control values to be given as U = [-control_size, control_size]
    controldim = 5
    perturbation = np.random.normal(scale=0.1, size=(10, 10))
    T_matrix = np.eye(10) + perturbation
    T_inv = np.linalg.inv(T_matrix)

    def __init__(self):
        self.constant_size = 4.0  # transform the state space from [-constant_size, constant_size]**inputdim to [-1,1]**inputdim

    def vf(self, x):
        """
        Define the vector field vf.
        
        Parameters:
        x: Tensor of shape (batch_size, inputdim) representing the state
        
        Returns:
        Array of shape (inputdim, batch_size) representing the vector field f(x)
        """
        x = np.array(np.matmul(x, self.T_matrix.T))
        
        y = [- self.constant_size * x[:,0] + self.constant_size * x[:,1] * self.constant_size * x[:,0] - 0.1 * np.power(x[:, 8], 2), 0 * x[:,1], 
             - self.constant_size * x[:,2] + self.constant_size * x[:,3] * self.constant_size * x[:,2] - 0.1 * np.power(x[:, 0], 2), 0 * x[:,3],
             - self.constant_size * x[:,4] + self.constant_size * x[:,5] * self.constant_size * x[:,4] + 0.1 * np.power(x[:, 6], 2), 0 * x[:,5],
             - self.constant_size * x[:,6] + self.constant_size * x[:,7] * self.constant_size * x[:,6], 0 * x[:,7],
             - self.constant_size * x[:,8] + self.constant_size * x[:,9] * self.constant_size * x[:,8], 0 * x[:,9] + 0.1 * np.power(x[:, 1], 2)]
        
        return np.matmul(self.T_inv, y) 
    

    def vg(self, x):
        """
        Define the control vector field vg.
        
        Parameters:
        x: Tensor of shape (batch_size, inputdim) representing the state
        
        Returns:
        Array of shape (inputdim, controldim, batch_size) representing the control vector field g(x)
        """
        x = np.array(np.matmul(x, self.T_matrix.T))

        x1 = x[:, 1]
        x3 = x[:, 3]
        x5 = x[:, 5]
        x7 = x[:, 7]
        x9 = x[:, 9]
        res = np.array([[0 * x1, - self.constant_size * x1, 0 * x1, 0 * x1, 0 * x1, 0 * x1, 0 * x1, 0 * x1, 0 * x1, 0 * x1],
                        [0 * x1, 0 * x1, 0 * x1, - self.constant_size * x3, 0 * x1, 0 * x1, 0 * x1, 0 * x1, 0 * x1, 0 * x1],
                        [0 * x1, 0 * x1, 0 * x1, 0 * x1, 0 * x1, - self.constant_size * x5, 0 * x1, 0 * x1, 0 * x1, 0 * x1],
                        [0 * x1, 0 * x1, 0 * x1, 0 * x1, 0 * x1, 0 * x1, 0 * x1,  -self.constant_size * x7, 0 * x1, 0 * x1],
                        [0 * x1, 0 * x1, 0 * x1, 0 * x1, 0 * x1, 0 * x1, 0 * x1, 0 * x1, 0 * x1, - self.constant_size * x9]
                        ])
 
        return np.einsum('mnj,ni->mij', res, self.T_inv.T)
    
    def f(self, x, u):
        """
        Define the right-hand side function f.
        
        Parameters:
        x: Tensor of shape (batch_size, inputdim) representing the state
        u: Tensor of shape (batch_size, controldim) representing the control 
        
        Returns:
        Tensor of shape (inputdim, batch_size) representing the r.h.s f(x,u)
        """
        x = np.array(np.matmul(x, self.T_matrix.T))

        y = [- self.constant_size * x[:,0] + self.constant_size * x[:,1] * self.constant_size * x[:,0] - 0.1 * np.power(x[:, 8], 2),- self.constant_size * x[:,1] * u[:, 0], 
             - self.constant_size * x[:,2] + self.constant_size * x[:,3] * self.constant_size * x[:,2] - 0.1 * np.power(x[:, 0], 2),- self.constant_size * x[:,3] * u[:, 1],
             - self.constant_size * x[:,4] + self.constant_size * x[:,5] * self.constant_size * x[:,4] + 0.1 * np.power(x[:, 6], 2),- self.constant_size * x[:,5] * u[:, 2],
             - self.constant_size * x[:,6] + self.constant_size * x[:,7] * self.constant_size * x[:,6],- self.constant_size * x[:,7] * u[:, 3],
             - self.constant_size * x[:,8] + self.constant_size * x[:,9] * self.constant_size * x[:,8],- self.constant_size * x[:,9] * u[:, 4] + 0.1 * np.power(x[:, 1], 2)]
        
        y = np.array(y)
        result = tf.convert_to_tensor(np.matmul(self.T_inv, y),  dtype=tf.float32)

        return result
    

class Pendulum:
    """
    Class representing a 2-dimensional pendulum example \dot{x} = f(x,u) = vf(x) + vg(x) u.
    """

    # Class constants
    title = '2-dimensional pendulum'
    inputdim = 2  # Dimension of the state space
    interval_size = 1  # Interval size for the state space cube [-interval_size, interval_size]**inputdim
    control_size = 1  # Control values space U = [-control_size, control_size]
    controldim = 1  # Dimension of the control space

    def vf(self, x):
        """
        Define the vector field vf.
        
        Parameters:
        x: Tensor of shape (batch_size, inputdim) representing the state
        
        Returns:
        Array of shape (inputdim, batch_size) representing the vector field f(x)
        """
        x1 = x[:, 0]
        x2 = x[:, 1]
        return np.array([x2, -x2 + tf.sin(x1)]) 

    def vg(self, x):
        """
        Define the control vector field vg.
        
        Parameters:
        x: Tensor of shape (batch_size, inputdim) representing the state
        
        Returns:
        Array of shape (inputdim, controldim, batch_size) representing the control vector field g(x)
        """
        x1 = x[:, 0]
        x2 = x[:, 1]
        return np.array([[0 * x1, 1 + 0 * x2]]) 
    
    def f(self,x,u):
        """
        Define the right-hand side function f.
        
        Parameters:
        x: Tensor of shape (batch_size, inputdim) representing the state
        u: Tensor of shape (batch_size, controldim) representing the control 
        
        Returns:
        Tensor of shape (inputdim, batch_size) representing the r.h.s f(x,u)
        """
        x1 = x[:, 0]
        x2 = x[:, 1]
        res = [x2, -x2 + tf.sin(x1)+u[:,0]]
        return tf.convert_to_tensor(res,  dtype=tf.float32)
