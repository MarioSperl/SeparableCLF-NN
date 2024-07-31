import tensorflow as tf 


# data object to store the errors within the training process
class TrainingProgress:
    '''sum denotes L1 norm and max the maximal errors'''

    def __init__(self):
        self.max_val_uloss_errors = []
        self.sum_val_uloss_errors = []
        self.max_uloss_errors = []
        self.sum_uloss_errors = []

        self.max_val_errors = []
        self.sum_val_errors = []
        self.sum_val_bloss_error = []
        self.sum_val_gloss_error = []
        self.max_trainingpoint_errors = []
        self.sum_training_errors = []

        self.argmax_training_error = []
        self.argmax_val_error = []

        self.finalepoch = 0



def calculate_DV_vg_norm(gradx_batch, vg_batch_train, controldim):
    """Computes the 1-norm of DV(x) * g(x) for each sample in the batch, cf. Section 5 in the paper.  

    Args:
        gradx (tf.Tensor): A 2D tensor of shape (batch_size, statedim) representing the gradients.
        vg_batch_train (tf.Tensor): A 3D tensor of shape (batch_size, statedim, controldim). 
        controldim (int): An integer representing the number of control dimensions.

    Returns:
        tf.Tensor: A 1D tensor of shape (batch_size,) containing the accumulated sum of absolute values
                   for each sample in the batch.

    """
    erg = 0
    for idx in range(controldim):
        matmul_result = tf.reduce_sum(gradx_batch * vg_batch_train[:, :, idx], axis=1)
        erg += tf.abs(matmul_result)
    return erg