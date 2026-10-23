# DL-LT02 — Getting Started with PyTorch

## Overview

This assignment introduces the fundamental operations of PyTorch tensors and basic automatic differentiation. It includes practical exercises on tensor creation, indexing, reshaping, mathematical operations, statistical functions, and gradient-based optimization.

## Topics Covered

- **Tensor Fundamentals:** Creating tensors and inspecting shape, dimensions, and data types.
- **Tensor Operations:** Random tensor generation, indexing, slicing, reshaping, and flattening.
- **Mathematical Operations:** Element-wise operations and matrix multiplication.
- **Statistical Functions:** Mean, standard deviation, and cumulative sum.
- **Automatic Differentiation:** Computing derivatives with PyTorch Autograd.
- **Gradient-Based Optimization:** Learning polynomial coefficients and approximating solutions to a system of equations.
- **Practice Exercises:** Applying tensor operations, derivatives, and optimization in combined exercises.

## Repository Structure

```text
02-pytorch-basics/
├── README.md
├── tensor_basics.py
├── autograd_gradient_descent.py
├── practice_exercises.py
└── report.docx
```

## Requirements

- Python 3
- PyTorch
- NumPy

Install the required libraries:

```bash
pip install torch numpy
```

## How to Run

Run each exercise file from the project directory:

```bash
python tensor_basics.py
python autograd_gradient_descent.py
python practice_exercises.py
```

## Learning Outcomes

After completing this assignment, I practiced how to:

- Create and inspect PyTorch tensors.
- Select tensor elements and modify tensor shapes.
- Perform element-wise and matrix operations.
- Calculate statistical values along specified dimensions.
- Use `requires_grad`, `backward()`, and `.grad` to compute derivatives.
- Apply an optimizer to update parameters using gradients.

## Assignment Information

- **Course:** Deep Learning
- **Exercise:** EX-LT02 — Getting Started with PyTorch
- **Institution:** Saigon University

## Academic Note

This repository documents practical exercises and learning notes based on the coursework. The reported outputs and observations are drawn from the original assignment report.
