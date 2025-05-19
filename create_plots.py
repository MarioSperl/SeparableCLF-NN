import pickle

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import os, sys, random


from settings import *
from auxiliary import * 


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


# Runge Kutta 4
def RK4(x0, u, h):
    """
    Perform one step of the fourth-order Runge-Kutta (RK4) method to solve ordinary differential equations.

    Args:
        x0 (tf.Tensor): The initial state.
        u (tf.Tensor): The applied control.
        h (float): The step size.

    Returns:
        tf.Tensor: The updated state after the RK4 step.

    """

    k1 = h * tf.cast(tf.stack(param.example.f(x0, u), axis=1), tf.float32)
    k2 = h * tf.cast(tf.stack(param.example.f(x0 + k1 / 2, u), axis=1), tf.float32)
    k3 = h * tf.cast(tf.stack(param.example.f(x0 + k2 / 2, u), axis=1), tf.float32)
    k4 = h * tf.cast(tf.stack(param.example.f(x0 + k3, u), axis=1), tf.float32)
    k = (k1 + 2 * k2 + 2 * k3 + k4) / 6
    xn = x0 + k

    return xn

##------------------------------------------------------------------------------
##### plot computed CLF
def CLF(model=None, axis1=0, axis2=1, dirname='', zmin=-10., zmax=10., numpoints = 100, plot_bounds = False):
    """
    Generate and plot the Control Lyapunov Function (CLF) and its directional derivative projected onto axis1 and axis2.

    Args:
        model: The neural network model used to compute the CLF. 
        axis1 (int): The first axis to plot. Default is 0.
        axis2 (int): The second axis to plot. Default is 1.
        dirname (str): Directory name for saving the plot. Default is an empty string.
        zmin (float): The minimum value for the z-axis in the plot. Default is -10.
        zmax (float): The maximum value for the z-axis in the plot. Default is 10.
        numpoints (int): Number of points to use in the mesh grid for plotting. Default is 100.
        plot_bounds (bool): Whether to plot upper and lower bounds on the CLF. Default is False.

    Returns:
        None: The function generates and saves a 3D plot as a PDF.

    """


    # define plotting range and mesh
    x = np.linspace(-param.example.interval_size, param.example.interval_size, numpoints)
    y = np.linspace(-param.example.interval_size, param.example.interval_size, numpoints)

    X, Y = np.meshgrid(x, y)

    s = X.shape

    Ze = np.zeros(s)
    Ze2 = np.zeros(s)
    Zp = np.zeros(s)
    Zu = np.zeros(s)
    Zl = np.zeros(s)
    Zg = np.zeros(s)
    DT = np.zeros((numpoints ** 2, param.example.inputdim))

    # convert mesh into point vector for which the model can be evaluated
    c = 0
    for i in range(s[0]):
        for j in range(s[1]):
            DT[c, axis1 - 1] = X[i, j]
            DT[c, axis2 - 1] = Y[i, j]

            c = c + 1

    if model is None:
        model = tf.keras.models.load_model(f'{in_directory}/Models/Epoch_{progress.finalepoch}/Network', compile=False)

    # evaluate model (= Lyapunov function values V)
    Ep = model.predict(DT)[:, 0]

    # convert point vector to tensor for evaluating x-derivative
    tDT = tf.convert_to_tensor(DT, dtype=tf.float32)

    # evaluate gradients DV of Lyapunov function
    with tf.GradientTape() as tape:
        tape.watch(tDT)
        ypm = model(tDT)[:, 0]
        grads = tape.gradient(ypm, tDT)

    # compute orbital derivative
    Ee = tf.reduce_sum(grads * tf.transpose(tf.convert_to_tensor(param.example.vf(DT), dtype=tf.float32)), axis=1) - \
         param.example.control_size * calculate_DV_vg_norm(grads, tf.transpose(
        tf.convert_to_tensor(param.example.vg(DT), dtype=tf.float32)), param.example.controldim)

    # compute upperbound
    Eu = bound_functions.upperbound(DT)

    # compute lowerbound
    El = bound_functions.lowerbound(DT)

    # compute gradient bound
    Eg = bound_functions.gradientbound(DT)

    # copy V and DVf values into plottable format
    c = 0
    lenc = len(Ee)
    for i in range(s[0]):
        c2 = i
        for j in range(s[1]):
            Ze[i, j] = Ee[c]
            Ze2[i, j] = Ee[c2]
            Zp[i, j] = Ep[c]
            Zl[i, j] = El[c]
            Zu[i, j] = Eu[c]
            Zg[i, j] = - Eg[c]
            c = c + 1
            c2 = (c2 + s[1]) % lenc


    ### plot the calculated values 

    # Create a 3D plot using Matplotlib
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=6, azim=99)
    fig.set_size_inches(36, 16) 

    # Define a custom colormap with masked values as transparent
    cmap = plt.get_cmap('viridis')  # You can change 'viridis' to any other colormap
    cmap.set_bad('none')  # Set masked values as transparent


    # Add labels and adjust the z-axis limits
    ax.set_xlabel(r'$x_{}$'.format(axis1), fontsize = 28, labelpad = 15)
    ax.set_ylabel(r'$x_{}$'.format(axis2), fontsize = 28, labelpad = 15)
    ax.set_zlabel(r'$W, DWf$', fontsize = 28, labelpad = 15)
    ax.set_zlim(zmin, zmax)

    plt.xticks(fontsize = 16)
    plt.yticks(fontsize = 16)
    ax.tick_params(axis = 'z', labelsize = 16)

    
    ax.set_xticks([-1, - 0.5, 0, 0.5, 1], fontsize = 16)
    ax.set_yticks([-1, -0.5,  0, 0.5, 1], fontsize = 16)


    # Bring the data for wireframe in correct shape 

    # Define the margin
    margin = 0.05 # Adjust the margin as needed

    # Create a mask for values within the range [1, 1 + margin]
    mask = (Ze <= zmin) & (Ze >= zmin - margin)

    # Set values within the range to 1
    Ze[mask] = zmin 

    # Create a mask for values above 1 + margin
    mask_above_margin = (Ze < zmin - margin)

    # Set values above 1 + margin to NaN (not a number)
    Ze[mask_above_margin] = np.nan


    ax.plot_wireframe(X, Y, Ze, rstride=5, cstride=5)

    ## lowerbound 
    mask2 = (Zu >= zmax) & (Zu <= zmax + margin)

    # Set values within the range to 1
    Zu[mask2] = zmax 

    # Create a mask for values above 1 + margin
    mask_above_margin2 = (Zu > zmax + margin)

    # Set values above 1 + margin to NaN (not a number)
    Zu[mask_above_margin2] = np.nan

    if plot_bounds:
        ax.plot_surface(X, Y, Zu, rstride=5, cstride=5, color='orange', alpha = 1)

        ax.plot_surface(X, Y, Zl, rstride=5, cstride=5, color='orange', alpha = 1)

    surface = ax.plot_surface(X, Y, Zp, cmap=cmap) 

    # Save the plot as a PDF
    pdf_filename = f'Plot/CLF{dirname}.pdf'
    plt.savefig(pdf_filename, format="pdf", transparent=True)

    return


def plot_multiple_V_trajectories(dirname='', num_initial_states=5, steps=2500, stepsize=0.01, domain_factor = 1):
    """
    Plot multiple V trajectories, each starting from a different randomly generated initial state.

    Parameters:
    - dirname (str): Directory name to save the plot.
    - num_initial_states (int): Number of randomly generated initial states.
    - steps (int): Number of steps for the trajectory simulation.
    - stepsize (float): Step size for numerical integration.
    - domain_factor: pick initial points from domain_factor * [-interval_size, interval_size]**input_dim
    """
    
    interval_size = param.example.interval_size
    input_dim = param.example.inputdim
    
    # Load the trained model
    model_path = f'{in_directory}/Models/Epoch_{progress.finalepoch}/Network'
    model = tf.keras.models.load_model(model_path, compile=False)
    
    def compute_V_value(x):
        return model(x, training=False)[:, 0]

    # Compute the optimal control
    def compute_control(gradx, x):
        u = []
        for idx in range(param.example.controldim):
            vg_x = tf.convert_to_tensor(tf.cast(tf.transpose(param.example.vg(x)[idx]), dtype=tf.float32), dtype=tf.float32)
            u.append(-param.example.control_size * tf.sign(tf.reduce_sum(gradx * vg_x, axis=1)))
        return tf.transpose(tf.convert_to_tensor(u, dtype=tf.float32)) 

    def simulate_trajectory(x_initial):
        x_trajectory = tf.convert_to_tensor(x_initial, dtype=tf.float32)
        V_values = [compute_V_value(x_trajectory)]
        for _ in range(steps):
            with tf.GradientTape() as tape:
                tape.watch(x_trajectory)
                ypm = compute_V_value(x_trajectory)
                gradx = tape.gradient(ypm, x_trajectory)
            control_input = compute_control(gradx, x_trajectory)
            x_trajectory = RK4(x_trajectory, control_input, stepsize)
            V_values.append(compute_V_value(x_trajectory))
        return V_values

    # Plot settings
    t_list = [stepsize * step for step in range(steps + 1)]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Simulate and plot trajectories for each initial state
    for i in range(num_initial_states):
        x_initial = domain_factor * np.random.uniform(-interval_size, interval_size, (1, input_dim)) 
        V_values = simulate_trajectory(x_initial)
        if num_initial_states <= 5:
            ax.plot(t_list, V_values, linestyle='-', linewidth=2.5, label=f'Trajectory {i+1}')
        else:
            ax.plot(t_list, V_values, linestyle='-', linewidth=2.5)
    

    # Plot settings
    ax.set_xlabel(r'$t$', fontsize=28, labelpad=15)
    ax.set_ylabel(r'$W_{\theta}(x(t))$', fontsize=28, labelpad=15)
    ax.grid(True, linestyle='-', alpha=0.5)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
    if num_initial_states <= 5:
        ax.legend(fontsize=16)

    pdf_filename = f'Plot/V_{dirname}.pdf'
    plt.savefig(pdf_filename, format='pdf', transparent=True)

    print(f"Plot saved as {pdf_filename}")


if __name__ == '__main__':
    myDir = os.getcwd()
    sys.path.append(myDir)

    seed_value = 42
    set_seeds(seed_value)

    param = Param()
    bound_functions = BoundaryFunctions(param)
    in_directory = 'Data'
    out_directory = 'Plot'

    filehandler = open(f'{in_directory}/TrainingProgress.pkl', 'rb')
    progress = pickle.load(filehandler)

    # if not os.path.exists(out_directory):
    #     os.mkdir(out_directory)
    # CLF(axis1=0, axis2=2, dirname=f'_{param.example.title}_test', zmin=-12, zmax = 12, numpoints=300, plot_bounds=False)

    plot_multiple_V_trajectories(dirname=f'_{param.example.title}_Trajectory', num_initial_states=20, steps=1500, stepsize=0.001, domain_factor=0.5)


    plt.show()

    print('Finished')






