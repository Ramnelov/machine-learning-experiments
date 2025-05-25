# Spark Multi-Layer Perceptron (MLP)

This project implements a **multi-layer perceptron (MLP)** using **PySpark** to approximate the sine function from noisy observations. The goal is thus to learn a parametric function such that

$$
y = f_\theta\left( x \right) + \epsilon,
$$

meaning that we wish to learn the sine function, and not the noise.

We construct a feedforward neural network with a single hidden layer. Given MSE as the loss function, the following backpropagation algorithm can be derived for a single sample:

- Forward propagation

  1. $\mathbf{q}^{(0)} = x$
  1. $\mathbf{z}^{(1)} = \mathbf{W}^{(1)}\mathbf{q}^{(0)} + \mathbf{b}^{(1)}$
  1. $\mathbf{q}^{(1)} = h\left(\mathbf{z}^{(1)}\right)$
  1. $z^{(2)} = \mathbf{W}^{(2)}\mathbf{z}^{(1)} + \mathbf{b}^{(2)}$
  1. $J(\mathbf{\theta}) = \left(y - z^{(2)}\right)^2$

- Backward propagation

  1. $dz^{(2)} = -2\left(y - z^{(2)}\right)$
  1. $d\mathbf{q}^{(1)} = \mathbf{W}^{(2)\top}dz^{(2)}$
  1. $d\mathbf{z}^{(1)} = d\mathbf{q}^{(1)} \circ h'\left(\mathbf{z}^{(1)}\right)$
  1. $d\mathbf{W}^{(2)} = dz^{(2)}\mathbf{q}^{(1)\top}$
  1. $db^{(2)} = dz^{(2)}$
  1. $d\mathbf{W}^{(1)} = d\mathbf{z}^{(1)}\mathbf{q}^{(0)\top}$
  1. $d\mathbf{b}^{(1)} = d\mathbf{z}^{(1)}$

> The operation $\circ$ denotes element-wise multiplication.

Once the gradient for each parameter is computed for each data point (done in parallel using map) we aggregate these with a reduction and perform weight updates according to:

- Parameter updating

  1. $\mathbf{W}^{(2)}_{t+1} = \mathbf{W}^{(2)}_{t} - \gamma d\mathbf{W}^{(2)}_{t}$
  1. $\mathbf{b}^{(2)}_{t+1} = \mathbf{b}^{(2)}_{t} - \gamma d\mathbf{b}^{(2)}_{t}$
  1. $\mathbf{W}^{(1)}_{t+1} = \mathbf{W}^{(1)}_{t} - \gamma d\mathbf{W}^{(1)}_{t}$
  1. $b^{(1)}_{t+1} = b^{(1)}_{t} - \gamma db^{(1)}_{t}$

All of these computations are done for a set number of iterations.
