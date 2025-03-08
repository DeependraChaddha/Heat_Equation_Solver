# Physics-Informed Neural Networks (PINNs) for Solving the 2D Heat Equation

This repository contains implementations of **Physics-Informed Neural Networks (PINNs)** for solving the **2D Heat Equation**. Two versions of the solver are provided:

1. **Steady-State 2D Heat Equation**
2. **Time-Dependent 2D Heat Equation**

## 1. Problem Formulation

### Steady-State Heat Equation:
The governing equation is:
$\[\nabla^2 u(x, y) = f(x,y)\]$
where \( u(x, y) \) represents the temperature distribution in a 2D domain and $\f(x,y)$ is a given function.

### Time-Dependent Heat Equation:
The equation is:
$\[\frac{\partial u}{\partial t} = \alpha \nabla^2 u(x, y)\]$
where $\( \alpha \)$ is the thermal diffusivity.

## 2. Implementation Details

- The PINN framework is used to approximate the solution by minimizing a loss function that enforces the governing PDE and boundary conditions.
- A neural network takes $\( (x, y) \)$ (and $\( t \)$ for the time-dependent case) as inputs and predicts $\( u(x, y) \)$.
- The loss function consists of:
  - **PDE residual loss** (ensuring the equation holds)
  - **Boundary condition loss** (enforcing boundary values)
  - **Initial condition loss** (for the time-dependent case)

## 3. Dependencies

Ensure you have the following dependencies installed:

```bash
pip install numpy torch matplotlib
