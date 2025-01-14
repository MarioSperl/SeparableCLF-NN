import tensorflow as tf 

import examples

class Param:
    """
    Parameter configuration class for setting up the network structure, training data,
    and training loop parameters.
    """

    def __init__(self):
        """Initialize parameters for network generation, training data, and training loop."""

        '''Choose example here!'''
        self.example = examples.CLF_10dim()
        
        # ----- Parameters for network generation ----
        self.separable_structure = True  # True for separable network structure, False for fully connected DNN
        self.layersize = 64  # Size of the hidden layers for fully connected NN
        self.layers = 1  # Number of hidden layers for fully connected NN

        self.subnum = 5 # Number of sublayers in the reduced network structure
        self.subdim = 2  # Dimension of the input of each sublayer
        self.sublayersize =  64 # Size of the sublayer

        # ----- Parameters for training data ---- 
        self.data_size = 200000  # Number of randomly generated points for training
        self.val_size = 200000  # Number of randomly generated points for validation
        self.test_size = 100000  # Number of randomly generated points for testing

        self.adaptive_grid_size = 100000  # Number of points used for verification within adaptive sampling

        self.batch_size = 64  # Size of each batch in the training algorithm
        self.valbatch_size = self.batch_size  # Batch size for validation set
        self.testbatch_size = self.batch_size  # Batch size for test set
        self.adaptive_gridbatch_size = self.batch_size  # Batch size for verification set

        # --- Parameters for training loop ----
        self.max_epochs = 30  # Maximal number of epochs until the training process is stopped
        self.min_epochs = 10  # Minimal number of epochs until the training process is stopped
        self.tol = 1e-5   #modified: Tolerance on L1 error of training; previously L \infty applied on validation data for stopping the training

        # Set coefficients of the comparison functions
        self.upper = 10  # Upper bound \alpha_2
        self.lower = 0.5 # Lower bound \alpha_1
        self.gradient = 0.01  # Gradient bound
        self.gradweight = 1.0  # Weight between bound loss and gradient loss in loss function

        self.zeroloss = True  # If True, adds W(0)^2 to the loss function 
        self.zeroloss_gradient = True  # If True, adds |DW(0)|^2 to the loss function
        self.adaptive_sampling = False  # Enable or disable variations of the unsupervised learning methods
        
        # Set adaptive sampling parameters
        # self.adaptive_gridpoints_per_dim = 20  # Grid points per dimension of the verification grid
        self.adaptive_radius = 0.01  # Radius of the ball where the new data is generated
        self.adaptive_max = 100  # The amount of the largest errors which are considered
        self.adaptive_generate = 100  # The amount of generated points per state
        self.adaptive_interval = 10  # Number of epochs until next adaptive sampling is applied

        # Enlarge the training domain (not validation and test) by the following scalar
        self.buffer = 1.01
    
    def validate_parameters(self):
        """Validate parameters to ensure they are within acceptable ranges."""
        assert self.layersize > 0, "Layer size must be positive."
        assert self.layers > 0, "Number of layers cannot positive."
        assert self.subnum > 0, "Number of sublayers must be positive."
        assert self.subdim > 0, "Dimension of sublayer input must be positive."
        assert self.sublayersize > 0, "Size of sublayer must be positive."
        assert self.data_size > 0, "Data size must be positive."
        assert self.val_size > 0, "Validation size must be positive."
        assert self.test_size > 0, "Test size must be positive."
        assert self.adaptive_grid_size > 0, "Adaptive grid size must be positive."
        assert self.batch_size > 0, "Batch size must be positive."
        assert self.max_epochs >= self.min_epochs, "Max epochs must be greater than or equal to min epochs."
        assert self.tol > 0, "Tolerance must be positive."
        assert self.upper > self.lower, "Upper bound must be greater than lower bound."
        assert self.gradient > 0, "Gradient bound must be positive."
        assert self.gradweight > 0, "Gradient weight must be positive."
        assert self.adaptive_gridpoints_per_dim > 0, "Adaptive grid points per dimension must be positive."
        assert self.adaptive_radius > 0, "Adaptive radius must be positive."
        assert self.adaptive_max > 0, "Adaptive max must be positive."
        assert self.adaptive_generate > 0, "Adaptive generate must be positive."
        assert self.adaptive_interval > 0, "Adaptive interval must be positive."
        assert self.buffer >= 1, "Buffer must be greater or equal than 1."


class BoundaryFunctions:
    """
    This class defines the comparison functions with the chosen hyperparameters.
    """

    def __init__(self, param):
        """
        Initialize the BoundaryFunctions class with hyperparameters.
        
        Parameters:
        param: An object containing the upper, lower, and gradient hyperparameters.
        """
        self.upper = param.upper
        self.lower = param.lower
        self.gradient = param.gradient

    def upperbound(self, x):
        """
        Define the upper bound K-function.
        
        Parameters:
        x: Tensor input for the upper bound function.
        
        Returns:
        Tensor representing the upper bound.
        """
        result = tf.reduce_sum(tf.square(x), axis=1)
        return self.upper * result

    def lowerbound(self, x):
        """
        Define the lower bound K-function.
        
        Parameters:
        x: Tensor input for the lower bound function.
        
        Returns:
        Tensor representing the lower bound.
        """
        result = tf.reduce_sum(tf.square(x), axis=1)
        return self.lower * result

    def gradientbound(self, x):
        """
        Define the function for the gradient condition.
        
        Parameters:
        x: Tensor input for the gradient condition function.
        
        Returns:
        Tensor representing the gradient bound.
        """
        result = tf.reduce_sum(tf.square(x), axis=1)
        return self.gradient * result