#  Approximation of Separable Control Lyapunov Functions with Neural Networks 

This repository contains Tensorflow code for approximating a control Lyapunov function (CLF) of a given control system using a feedforward neural network. The code is based on the investigation in the paper ["Approximation of Separable Control Lyapunov Functions with Neural Networks"](https://eref.uni-bayreuth.de/id/eprint/87696/).

## Repository Structure

The repository consists of the following Python files:

- **examples.py**: Defines several control systems for which Control Lyapunov Functions are to be learned.
- **settings.py**: Loads a particular example from `examples.py` and defines all necessary parameters for training.
- **main.py**: Performs the training process and saves the trained model.
- **auxiliary.py**: Contains auxiliary methods used throughout the project.
- **create_plots.py**: (Optional) This script can be run to visualize the results of the training process. It loads the last trained model and generates a PDF plot based on the selected visualization method.

## Installation

To install the required dependencies, use the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Usage

1. Define the control system in `examples.py`.
2. Configure training parameters in `settings.py`.
3. Run `main.py` to train the model.
4. (Optional) Run `create_plots.py` to visualize the training results.

