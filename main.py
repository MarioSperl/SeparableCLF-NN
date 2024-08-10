#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ------------------------------------------------------------------------------
#### Imports 

import os, sys
import time
import random 

import numpy as np 

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle

from settings import *
from auxiliary import * 


# import numpy as np
# import matplotlib.pyplot as plt
# import sys
# import os
# import tensorflow as tf 
# from numpy.random import default_rng
# import itertools

# sys.path.insert(1, os.getcwd())


# ------------------------------------------------------------------------------
#### Standalone Functions  

def set_seeds(seed_value):
    """
    Set random seeds for reproducibility.

    Args:
    seed_value: Integer seed value for reproducibility.
    """
    # Set the seed for NumPy
    np.random.seed(seed_value)

    # Set the seed for random module
    random.seed(seed_value)

    # Set the seed for TensorFlow
    tf.random.set_seed(seed_value)

    # If using GPU, ensure deterministic operations
    # Note: This may impact performance
    if tf.config.list_physical_devices('GPU'):
        # Disable GPU's non-deterministic operations
        tf.config.experimental.enable_op_determinism()

@tf.function
def bound_loss_vec(y_pred, ubound, lbound, zdata):
    """
    Define the bound loss function using upper and lower bounds.
    
    Args:
    y_pred: Tensor of NN values
    ubound: Tensor of upper bound values
    lbound: Tensor of lower bound values
    zdata: Tensor of zero data values
    
    Returns:
    Tensor representing the custom loss.
    """
    lower_loss = tf.square(tf.reduce_min(tf.stack([y_pred - lbound, zdata], axis=0), axis=0))
    upper_loss = tf.square(tf.reduce_max(tf.stack([y_pred - ubound, zdata], axis=0), axis=0))
    custom_loss = lower_loss + upper_loss
    return custom_loss


@tf.function
def grad_loss_le_vec_affin(gradx, x_batch_train, vf_batch_train, vg_batch_train, zeros_batch_train):
    """
    Computes the gradient loss for nonlinear systems \dot{x} = f(x) + g(x) u with bounded control values.

    Args:
    gradx: Tensor representing the gradient of V w.r.t. x 
    x_batch_train: Tensor of input batch training data
    vf_batch_train: Tensor of f evaluated at batch training data
    vg_batch_train: Tensor of g evaluated batch training data
    zeros_batch_train: Tensor with zeros for loss calculation

    Returns:
    Tensor representing the gradient loss.
    """
    # Compute the differential term
    diff = tf.reduce_sum(gradx * vf_batch_train, axis=1) + \
           bound_functions.gradientbound(x_batch_train) - \
           param.example.control_size * calculate_DV_vg_norm(gradx, vg_batch_train, param.example.controldim)
    
    # Compute the gradient loss
    g_loss = tf.square(tf.reduce_max([diff, zeros_batch_train], 0))
    
    return g_loss


@tf.function
def train_batch(dnn, x_batch_train, ub_batch_train, lb_batch_train, vf_batch_train, vg_batch_train, zeros_batch_train):
    """
    Performs optimization for one batch of data.

    Args:
    dnn: An instance of class DNN 
    x_batch_train: Tensor containing the input batch for training.
    ub_batch_train: Tensor containing the upper bound values for the loss function.
    lb_batch_train: Tensor containing the lower bound values for the loss function.
    vf_batch_train: Tensor containing f(x) for the batch training data.
    vg_batch_train: Tensor containing g(x) for the batch training data.
    zeros_batch_train: Tensor representing zeros for loss calculations.

    Returns:
    A tuple containing:
    - loss_value_vec: Tensor of the combined loss values.
    - bloss_vec: Tensor of boundary condition loss values.
    - gloss_vec: Tensor of gradient loss values.
    """

    # Start gradient recording for model parameters
    with tf.GradientTape() as tape:
        # Evaluate model
        logits = dnn.model(x_batch_train, training=True)

        # Evaluate boundary condition part of loss function
        bloss_vec = bound_loss_vec(logits[:, 0], ub_batch_train, lb_batch_train, zeros_batch_train)
        bloss = tf.reduce_sum(bloss_vec)

        # Start gradient recording for derivative w.r.t. x
        with tf.GradientTape() as tapex, tf.GradientTape() as tape0:
            # Watch the input batch
            tapex.watch(x_batch_train)
            logits2 = dnn.model(x_batch_train)

            # Evaluate x-derivative
            gradx = tapex.gradient(logits2, x_batch_train)

            # Evaluate PDE part of loss function
            gloss_vec = grad_loss_le_vec_affin(gradx, x_batch_train, vf_batch_train, vg_batch_train, zeros_batch_train)
            gloss = tf.reduce_sum(gloss_vec)

            # Calculate combined loss values
            loss_value_vec = bloss_vec + param.gradweight * gloss_vec
            loss_value = bloss + param.gradweight * gloss

            zeros = tf.zeros_like(x_batch_train)
            zero_loss_vec = zeros 
            zero_grad_loss_vec = zeros 
            # If true, add W(0)^2 to the loss function
            if param.zeroloss:
                if param.zeroloss_gradient:
                    tape0.watch(zeros)

                W_zero = dnn.model(zeros)
                zero_loss_vec = tf.square(W_zero[:, 0])
                loss_value_vec += zero_loss_vec
                loss_value += tf.reduce_sum(zero_loss_vec)

                if param.zeroloss_gradient:
                    grad_zero = tape0.gradient(W_zero, zeros)  # Calculate gradient of W_zero w.r.t. input
                    zero_grad_loss_vec = tf.reduce_sum(tf.square(grad_zero), axis = 1)
                    loss_value_vec += zero_grad_loss_vec
                    loss_value += tf.reduce_sum(tf.square(grad_zero), axis = 0)

        # Evaluate derivative w.r.t. model parameters
        grads = tape.gradient(loss_value, dnn.model.trainable_weights)

    # Run one step of gradient descent optimizer
    dnn.optimizer.apply_gradients(zip(grads, dnn.model.trainable_weights))

    return loss_value_vec, bloss_vec, gloss_vec, zero_loss_vec, zero_grad_loss_vec


def Training(model, data, progress, param):
    """
    Train the deep neural network model with data from `data` and track performance metrics.

    Args:
        model: The deep neural network model to be trained.
        data: An instance from the class Data containing training and validation datasets and methods.
        progress: An instance of the class Trainingprogress to store and track training and validation errors over epochs.
        param: An instance of the class Param containing hyperparameters and configuration settings for training.
    
    Returns:
        None
    """
    tolerance_reached = False

    # Loop through the epochs
    for epoch in range(param.max_epochs):
        # Prepare dataset
        train_dataset = data.train_dataset_raw.shuffle(buffer_size=1024).batch(param.batch_size)

        # Initialize error variables
        mlv_single = 0.0  # Maximum (L_infty) at one evaluation point
        slv = 0.0  # L1
        blv = 0.0  # L1 for bound loss
        glv = 0.0  # L1 for gradient loss
        zlv = 0.0  # Zero-loss 
        zglv = 0.0 # Zero-gradient loss 

        # Iterate over the batches of the dataset
        for step, (
                x_batch_train, ub_batch_train, lb_batch_train, vf_batch_train, vg_batch_train,
                zeros_batch_train) in enumerate(train_dataset):
            
            # Call optimization routine
            loss_value_vec, bloss_vec, gloss_vec, zero_loss_vec, zero_grad_loss_vec = train_batch(
                model, x_batch_train, ub_batch_train, lb_batch_train, vf_batch_train, vg_batch_train, zeros_batch_train)

            # Get current errors
            current_sum_loss = tf.reduce_sum(loss_value_vec)
            current_max_loss = tf.reduce_max(loss_value_vec)
            current_sum_bloss = tf.reduce_sum(bloss_vec)
            current_sum_gloss = tf.reduce_sum(gloss_vec)
            current_zero_loss = tf.reduce_sum(zero_loss_vec)
            current_zero_grad_loss = tf.reduce_sum(zero_grad_loss_vec)

            # Update errors
            mlv_single = tf.reduce_max([mlv_single, current_max_loss])
            slv += current_sum_loss
            blv += current_sum_bloss
            glv += current_sum_gloss
            zlv += current_zero_loss
            zglv += current_zero_grad_loss
            progress.argmax_training_error.append(x_batch_train[tf.argmax(loss_value_vec)])


        print(f'---------- epoch: {epoch:2d}')
        print(
            f'Training  : samples {((step + 1) * param.batch_size):8d}, '
            f'max-loss (single) {mlv_single:.6e}, '
            f'L1-loss {slv / (data.train_samples_per_epoch):.6e}, '
            f'boundary L1-loss {blv / (data.train_samples_per_epoch):.6e}, '
            f'gradient L1-loss {glv / (data.train_samples_per_epoch):.6e}, '
            f'loss at zero {zlv / (data.train_samples_per_epoch):.6e}, '
            f'gradient loss at zero {zglv / (data.train_samples_per_epoch):.6e} '
        )

        # Validation
        max_loss_val = 0
        sum_loss_val = 0
        val_dataset = data.val_dataset_raw.shuffle(buffer_size=1024).batch(param.valbatch_size)
        loss_val_vec_all = tf.zeros([0], dtype=tf.float32)
        val_data_all = tf.zeros([0, param.example.inputdim], dtype=tf.float32)

        for vdata_tf, val_ubound_tf, val_lbound_tf, val_vf_tf, val_vg_tf, val_zeros_tf in val_dataset:
            with tf.GradientTape() as tapeVal:
                tapeVal.watch(vdata_tf)
                logits_val = model.model(vdata_tf, training=False)

                bloss_val_vec = bound_loss_vec(logits_val[:, 0], val_ubound_tf, val_lbound_tf, val_zeros_tf)
                bloss_val = tf.reduce_sum(bloss_val_vec)

                gradx_val = tapeVal.gradient(logits_val, vdata_tf)

                gloss_val_vec = grad_loss_le_vec_affin(gradx_val, vdata_tf, val_vf_tf, val_vg_tf, val_zeros_tf)
                gloss_val = tf.reduce_sum(gloss_val_vec)

                loss_val_vec = bloss_val_vec + param.gradweight * gloss_val_vec
                max_loss_val = tf.reduce_max([tf.reduce_max(loss_val_vec), max_loss_val])
                sum_loss_val += bloss_val + param.gradweight * gloss_val

                loss_val_vec_all = tf.concat([loss_val_vec_all, loss_val_vec], axis=0)
                val_data_all = tf.concat([val_data_all, vdata_tf], axis=0)
        
        if epoch == 0: 
            print('Note that validation and test data in contrast to training never incorporate the terms V(0)^2 + || DV(0) ||_2^2 ')

        print(
            f'Validation: samples {param.val_size:8d}, '
            f'max-loss (single) {max_loss_val:.6e}, '
            f'L1-loss {(sum_loss_val / data.val_samples_per_epoch):.6e} '
        )

        # Adaptive sampling
        if param.adaptive_sampling and epoch > 0:
            if epoch % param.adaptive_interval == 0:
                loss_grid_vec_all = tf.zeros([0], dtype=tf.float32)
                grid_data_all = tf.zeros([0, param.example.inputdim], dtype=tf.float32)

                grid_dataset = data.grid_dataset_raw.shuffle(buffer_size=1024).batch(param.adaptive_gridbatch_size)
                for gdata_tf, grid_ubound_tf, grid_lbound_tf, grid_vf_tf, grid_vg_tf, grid_zeros_tf in grid_dataset:
                    with tf.GradientTape() as tapex:
                        tapex.watch(gdata_tf)
                        logits1_grid = model.model(gdata_tf, training=False)

                        bloss_grid_vec = bound_loss_vec(logits1_grid[:, 0], grid_ubound_tf, grid_lbound_tf, grid_zeros_tf)
                        gradx_grid = tapex.gradient(logits1_grid, gdata_tf)
                        gloss_grid_vec = grad_loss_le_vec_affin(gradx_grid, gdata_tf, grid_vf_tf, grid_vg_tf, grid_zeros_tf)

                        loss_grid_vec = bloss_grid_vec + param.gradweight * gloss_grid_vec

                        loss_grid_vec_all = tf.concat([loss_grid_vec_all, loss_grid_vec], axis=0)
                        grid_data_all = tf.concat([grid_data_all, gdata_tf], axis=0)

                partition_indices = np.argpartition(loss_grid_vec_all.numpy(), -param.adaptive_max)[-param.adaptive_max:]

                new_points = []
                for index in partition_indices:
                    for _ in range(param.adaptive_generate):
                        grid_point = grid_data_all[index]

                        random_vector = np.random.uniform(-1, 1, size=param.example.inputdim)
                        normalized_random_vector = random_vector / np.linalg.norm(random_vector)
                        scaling_factor = np.random.uniform(0, 1)

                        new_point = grid_point + normalized_random_vector * scaling_factor * param.adaptive_radius
                        new_point = new_point - 2 * tf.reduce_max([new_point - data.bound_x_tf, 0 * new_point], 0)
                        new_point = new_point - 2 * tf.reduce_min([new_point + data.bound_x_tf, 0 * new_point], 0)

                        new_points.append(new_point)

                new_points = np.array(new_points)
                param.data_size += new_points.shape[0]
                data.adaptive_extend_dataset(new_points)
                

        # Update errors
        progress.max_val_errors.append(max_loss_val)
        progress.sum_val_errors.append(sum_loss_val / data.val_samples_per_epoch)
        progress.max_trainingpoint_errors.append(mlv_single)
        progress.sum_training_errors.append(slv / data.train_samples_per_epoch)
        progress.sum_val_bloss_error.append(blv / data.train_samples_per_epoch)
        progress.sum_val_gloss_error.append(glv / data.train_samples_per_epoch)

        if epoch > (param.min_epochs - 2):
            if max_loss_val < param.tol:
                print('--------------------')
                print(f'training process stopped at epoch {epoch} with a max-error of {max_loss_val:e} and a tolerance of {param.tol:e}')
                tolerance_reached = True
                progress.finalepoch = epoch + 1
                break

    if not tolerance_reached:
        print('--------------------')
        print(f'Desired Tolerance was not reached')
        print(f'Training was stopped after {param.max_epochs:2d} epochs')
        progress.finalepoch = param.max_epochs

    model.save_model(progress.finalepoch)

    time2 = time.perf_counter()
    timediff = time2 - model.time1
    print(f'time for learning: {timediff:.2f}s')


def evaluate_TestData(model, data, param):
    """
    Evaluate the model's performance on the test dataset.

    Args:
        model: The trained deep neural network model.
        data: An object containing the test dataset.
        param: An object containing hyperparameters and configuration settings.
    
    Returns:
        None
    """
    max_loss_test = 0
    sum_loss_test = 0

    test_dataset = data.test_dataset_raw.shuffle(buffer_size=1024).batch(param.testbatch_size)

    for testdata_tf, test_ubound_tf, test_lbound_tf, test_vf_tf, test_vg_tf, test_zeros_tf in test_dataset:
        with tf.GradientTape() as tapex:
            tapex.watch(testdata_tf)
            logits1_test = model.model(testdata_tf, training=False)

            bloss_test_vec = bound_loss_vec(logits1_test[:, 0], test_ubound_tf, test_lbound_tf, test_zeros_tf)
            bloss_test = tf.reduce_sum(bloss_test_vec)

            gradx_test = tapex.gradient(logits1_test, testdata_tf)

            gloss_test_vec = grad_loss_le_vec_affin(gradx_test, testdata_tf, test_vf_tf, test_vg_tf, test_zeros_tf)
            gloss_test = tf.reduce_sum(gloss_test_vec)

            loss_test_vec = bloss_test_vec + param.gradweight * gloss_test_vec
            max_loss_test = tf.reduce_max([tf.reduce_max(loss_test_vec), max_loss_test])
            sum_loss_test += bloss_test + param.gradweight * gloss_test

    print('evaluation at test data: max-loss {:e}, L1-loss {:e}'.format(
        float(max_loss_test), float(sum_loss_test / data.test_samples_per_epoch)))


# ------------------------------------------------------------------------------
#### Classes   

class DNN:
    """
    Construct and manage the neural network model.
    """
    def __init__(self):
        # Store time at the beginning of the computation
        self.time1 = time.perf_counter()

        # Initialize the model based on separable structure parameter
        self.model = self._build_model()
        
        # Compile the model
        self.optimizer = keras.optimizers.Adam()
        self.model.compile(optimizer=self.optimizer)
        
        # Print model summary
        self.model.summary()

    def _build_model(self):
        """
        Build the model based on whether a separable structure is used.
        
        Returns:
            model: The constructed Keras model.
        """
        if param.separable_structure:
            return self._build_separable_model()
        else:
            return self._build_standard_model()

    def _build_separable_model(self):
        """
        Build a separable structured model.
        
        Returns:
            model: The constructed Keras model with separable structure.
        """
        inputs = keras.Input(shape=(param.example.inputdim,), name='state')

        # Coordinate transformation layers
        xc = [layers.Dense(param.subdim, activation='linear', name=f'coord_trafo{i}')(inputs) 
              for i in range(param.subnum)]

        # Subsystem layers
        xs = [layers.Dense(param.sublayersize, activation='softplus', name=f'subsystem_{i}')(xc[i]) 
              for i in range(param.subnum)]

        # Concatenate subsystems and define output layer
        x = layers.concatenate(xs)
        output = layers.Dense(1, activation='linear', name='Lyapunov_function')(x)

        return keras.Model(inputs=inputs, outputs=output)

    def _build_standard_model(self):
        """
        Build a standard structured model.
        
        Returns:
            model: The constructed Keras model with standard structure.
        """
        inputs = keras.Input(shape=(param.example.inputdim,), name='state')

        # Hidden layers
        x = layers.Dense(param.layersize, activation='softplus', name='Hidden_Layer_1_Model')(inputs)
        for i in range(1, param.layers):
            x = layers.Dense(param.layersize, activation='softplus', name=f'Hidden_Layer_{i + 1}_Model')(x)
        
        # Output layer
        output = layers.Dense(1, activation='linear', name='Lyapunov_function')(x)

        return keras.Model(inputs=inputs, outputs=output)

    def save_model(self, epoch):
        """
        Save the model at a given epoch.
        
        Args:
            epoch (int): The current epoch number.
        
        Returns:
            None
        """
        model_dir = os.path.join(directory, 'Models', f'Epoch_{epoch}', 'Network')
        self.model.save(model_dir)

class Data:
    """
    Generate and manage training, validation, and testing data for the neural network model.
    """
    def __init__(self, param):
        """
        Initialize the Data object by generating the training, validation, and test datasets.
        
        Args:
            param: An object containing various parameters for data generation.
        """
        self._generate_training_data(param)
        self._generate_validation_data(param)
        self._generate_test_data(param)
        
        self.train_samples_per_epoch = param.data_size + param.data_size % param.batch_size
        self.val_samples_per_epoch = param.val_size + param.val_size % param.valbatch_size
        self.test_samples_per_epoch = param.test_size + param.test_size % param.testbatch_size                                  
        self.adaptive_samples_per_epoch = param.adaptive_grid_size + param.adaptive_grid_size % param.adaptive_gridbatch_size
    
    def _generate_training_data(self, param):
        """
        Generate training data and associated tensors.
        
        Args:
            param: An object containing various parameters for data generation.
        """
        self.tdata = 2 * param.buffer * param.example.interval_size * np.random.random(
            (param.data_size, param.example.inputdim)) - param.example.interval_size * param.buffer

        self.t_ubound = tf.cast(bound_functions.upperbound(self.tdata), tf.float32)
        self.t_lbound = tf.cast(bound_functions.lowerbound(self.tdata), tf.float32)
        self.vf_tdata = param.example.vf(self.tdata)
        self.vg_tdata = param.example.vg(self.tdata)

        self.tdata_tf = tf.convert_to_tensor(self.tdata, dtype=tf.float32)
        self.t_ubound_tf = tf.convert_to_tensor(self.t_ubound, dtype=tf.float32)
        self.t_lbound_tf = tf.convert_to_tensor(self.t_lbound, dtype=tf.float32)
        self.t_vf_tf = tf.transpose(tf.convert_to_tensor(self.vf_tdata, dtype=tf.float32))
        self.t_vg_tf = tf.transpose(tf.convert_to_tensor(self.vg_tdata, dtype=tf.float32))
        self.t_zeros_tf = tf.zeros(self.t_ubound_tf.shape)

        self.train_dataset_raw = tf.data.Dataset.from_tensor_slices(
            (self.tdata_tf, self.t_ubound_tf, self.t_lbound_tf, self.t_vf_tf, self.t_vg_tf, self.t_zeros_tf))

        if param.adaptive_sampling:
            self._generate_adaptive_data(param)

    def _generate_adaptive_data(self, param):
        """
        Generate adaptive grid data for sampling.
        
        Args:
            param: An object containing various parameters for data generation.
        """
        self.adaptive_gdata = 2 * param.example.interval_size * np.random.random(
            (param.adaptive_grid_size, param.example.inputdim)) - param.example.interval_size

        self.bound_gamma_tf = param.adaptive_radius * tf.ones([param.example.inputdim], dtype=tf.float32)
        self.bound_x_tf = param.example.interval_size * tf.ones([param.example.inputdim], dtype=tf.float32)
        self.gdata_tf = tf.convert_to_tensor(self.adaptive_gdata, dtype=tf.float32)

        self.grid_ubound = tf.cast(bound_functions.upperbound(self.adaptive_gdata), tf.float32)
        self.grid_lbound = tf.cast(bound_functions.lowerbound(self.adaptive_gdata), tf.float32)
        self.grid_ubound_tf = tf.convert_to_tensor(self.grid_ubound, dtype=tf.float32)
        self.grid_lbound_tf = tf.convert_to_tensor(self.grid_lbound, dtype=tf.float32)

        self.vf_gdata = param.example.vf(self.adaptive_gdata)
        self.vg_gdata = param.example.vg(self.adaptive_gdata)
        self.grid_vf_tf = tf.transpose(tf.convert_to_tensor(self.vf_gdata, dtype=tf.float32))
        self.grid_vg_tf = tf.transpose(tf.convert_to_tensor(self.vg_gdata, dtype=tf.float32))

        self.grid_zeros_tf = tf.zeros(self.grid_ubound_tf.shape)

        self.grid_dataset_raw = tf.data.Dataset.from_tensor_slices(
            (self.gdata_tf, self.grid_ubound_tf, self.grid_lbound_tf, self.grid_vf_tf, self.grid_vg_tf, self.grid_zeros_tf))

    def _generate_validation_data(self, param):
        """
        Generate validation data and associated tensors.
        
        Args:
            param: An object containing various parameters for data generation.
        """
        self.valdata = 2 * param.example.interval_size * np.random.random(
            (param.val_size, param.example.inputdim)) - param.example.interval_size

        self.val_ubound = tf.cast(bound_functions.upperbound(self.valdata), tf.float32)
        self.val_lbound = tf.cast(bound_functions.lowerbound(self.valdata), tf.float32)
        self.vf_vdata = param.example.vf(self.valdata)
        self.vg_vdata = param.example.vg(self.valdata)

        self.vdata_tf = tf.convert_to_tensor(self.valdata, dtype=tf.float32)
        self.val_ubound_tf = tf.convert_to_tensor(self.val_ubound, dtype=tf.float32)
        self.val_lbound_tf = tf.convert_to_tensor(self.val_lbound, dtype=tf.float32)
        self.val_vf_tf = tf.transpose(tf.convert_to_tensor(self.vf_vdata, dtype=tf.float32))
        self.val_vg_tf = tf.transpose(tf.convert_to_tensor(self.vg_vdata, dtype=tf.float32))
        self.val_zeros_tf = tf.zeros(self.val_ubound_tf.shape)

        self.val_dataset_raw = tf.data.Dataset.from_tensor_slices(
            (self.vdata_tf, self.val_ubound_tf, self.val_lbound_tf, self.val_vf_tf, self.val_vg_tf, self.val_zeros_tf))

    def _generate_test_data(self, param):
        """
        Generate test data and associated tensors.
        
        Args:
            param: An object containing various parameters for data generation.
        """
        self.testdata = 2 * param.example.interval_size * np.random.random(
            (param.test_size, param.example.inputdim)) - param.example.interval_size

        self.test_ubound = tf.cast(bound_functions.upperbound(self.testdata), tf.float32)
        self.test_lbound = tf.cast(bound_functions.lowerbound(self.testdata), tf.float32)
        self.vf_testdata = param.example.vf(self.testdata)
        self.vg_testdata = param.example.vg(self.testdata)

        self.testdata_tf = tf.convert_to_tensor(self.testdata, dtype=tf.float32)
        self.test_ubound_tf = tf.convert_to_tensor(self.test_ubound, dtype=tf.float32)
        self.test_lbound_tf = tf.convert_to_tensor(self.test_lbound, dtype=tf.float32)
        self.test_vf_tf = tf.transpose(tf.convert_to_tensor(self.vf_testdata, dtype=tf.float32))
        self.test_vg_tf = tf.transpose(tf.convert_to_tensor(self.vg_testdata, dtype=tf.float32))
        self.test_zeros_tf = tf.zeros(self.test_ubound_tf.shape)

        self.test_dataset_raw = tf.data.Dataset.from_tensor_slices(
            (self.testdata_tf, self.test_ubound_tf, self.test_lbound_tf, self.test_vf_tf, self.test_vg_tf, self.test_zeros_tf))
    
    def adaptive_extend_dataset(self, new_data):
        """
        Extend the existing training dataset with newly generated states from adaptive sampling.
        
        Args:
            new_data: Newly generated data points to be added to the training dataset.
        """
        initial_size = self.tdata_tf.shape[0]

        self.t_ubound = tf.concat([self.t_ubound, tf.cast(bound_functions.upperbound(new_data), tf.float32)], axis=0)
        self.t_lbound = tf.concat([self.t_lbound, tf.cast(bound_functions.lowerbound(new_data), tf.float32)], axis=0)

        self.vf_tdata_new = param.example.vf(new_data)
        self.t_vf_tf_new = tf.transpose(tf.convert_to_tensor(self.vf_tdata_new, dtype=tf.float32))
        self.t_vf_tf = tf.concat([self.t_vf_tf, self.t_vf_tf_new], axis=0)

        self.vg_tdata_new = param.example.vg(new_data)
        self.t_vg_tf_new = tf.transpose(tf.convert_to_tensor(self.vg_tdata_new, dtype=tf.float32))
        self.t_vg_tf = tf.concat([self.t_vg_tf, self.t_vg_tf_new], axis=0)

        self.tdata_tf = tf.concat([self.tdata_tf, tf.convert_to_tensor(new_data, dtype=tf.float32)], axis=0)
        self.t_ubound_tf = tf.convert_to_tensor(self.t_ubound, dtype=tf.float32)
        self.t_lbound_tf = tf.convert_to_tensor(self.t_lbound, dtype=tf.float32)

        self.t_zeros_tf = tf.zeros(self.t_ubound_tf.shape)
        self.train_dataset_raw = tf.data.Dataset.from_tensor_slices(
            (self.tdata_tf, self.t_ubound_tf, self.t_lbound_tf, self.t_vf_tf, self.t_vg_tf, self.t_zeros_tf))
        
        added_size = self.tdata_tf.shape[0] - initial_size
        print(f"Adaptive sampling: {added_size} points have been added to the training dataset.")
        

if __name__ == '__main__':
    # Set the current directory and append to sys.path
    myDir = os.getcwd()
    sys.path.append(myDir)

    # Set seed for reproducibility
    seed_value = 42
    set_seeds(seed_value)

    # Initialize parameters, directories, and data objects
    param = Param()
    directory = 'Data'

    bound_functions = BoundaryFunctions(param)
    data = Data(param)

    # Initialize and train the model
    network = DNN()
    progress = TrainingProgress()
    Training(network, data, progress, param)
    evaluate_TestData(network, data, param)

    # Save training progress
    filename = os.path.join(directory, 'TrainingProgress.pkl')

    if os.path.exists(filename):
        os.remove(filename)
    
    with open(filename, 'wb') as filehandler:
        pickle.dump(progress, filehandler)
