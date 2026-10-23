
import torch
import numpy as np


def exercise_11a():
    """Create and reshape a tensor containing values from 1 to 24."""
    print("=== Exercise 11a: Tensor Creation and Reshape ===")

    a = torch.arange(1, 25)
    b = a.reshape(2, 3, 4)

    print("Original tensor:")
    print(a)
    print("\nReshaped tensor (2, 3, 4):")
    print(b)
    print("Shape:", b.shape)
    print("Number of dimensions:", b.ndim)


def exercise_11b():
    """Perform elementwise arithmetic and statistical operations."""
    print("\n=== Exercise 11b: Tensor Operations ===")

    a = torch.arange(1, 13, dtype=torch.float32).reshape(3, 4)
    b = torch.full((3, 4), 2.0)

    print("Tensor A:\n", a)
    print("Tensor B:\n", b)
    print("A + B:\n", a + b)
    print("A - B:\n", a - b)
    print("A * B:\n", a * b)
    print("A / B:\n", a / b)
    print("Mean along dimension 0:\n", a.mean(dim=0))
    print("Standard deviation along dimension 0:\n", a.std(dim=0))


def exercise_11c():
    """Calculate the derivative using automatic differentiation."""
    print("\n=== Exercise 11c: Derivative with Autograd ===")

    x = torch.tensor(2.0, requires_grad=True)
    y = 2 * x**2 + 5 * x + 3

    y.backward()

    print("Function: y = 2*x^2 + 5*x + 3")
    print("x =", x.item())
    print("y =", y.item())
    print("dy/dx =", x.grad.item())


def exercise_11d():
    """Fit a quadratic polynomial using gradient descent."""
    print("\n=== Exercise 11d: Polynomial Fitting ===")

    # Target function: y = 2*x^2 - 3*x + 5
    np.random.seed(42)
    x = np.linspace(-3, 3, 100)
    y = 2 * x**2 - 3 * x + 5

    xx = torch.tensor(
        np.column_stack((x**2, x, np.ones_like(x))),
        dtype=torch.float32,
    )
    yy = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

    torch.manual_seed(42)
    w = torch.randn(3, 1, requires_grad=True)
    optimizer = torch.optim.NAdam([w], lr=0.01)

    for epoch in range(1000):
        optimizer.zero_grad()
        predictions = xx @ w
        loss = torch.mean((yy - predictions) ** 2)
        loss.backward()
        optimizer.step()

    coefficients = w.detach().flatten()

    print("Target coefficients [a, b, c]: [2, -3, 5]")
    print("Learned coefficients [a, b, c]:")
    print(coefficients)
    print("Final MSE:", loss.item())


if __name__ == "__main__":
    exercise_11a()
    exercise_11b()
    exercise_11c()
    exercise_11d()
