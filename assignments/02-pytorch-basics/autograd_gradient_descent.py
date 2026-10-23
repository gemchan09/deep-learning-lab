
import numpy as np
import torch
import torch.nn as nn


def exercise_8():
    """Exercise 8: Automatic differentiation with Autograd."""
    print("=== Exercise 8: Autograd ===")

    x = torch.tensor(3.6, requires_grad=True)
    y = x * x
    y.backward()

    print("x =", x.item())
    print("y = x * x =", y.item())
    print("dy/dx =", x.grad.item())

    x2 = torch.tensor(2.0, requires_grad=True)
    y2 = 3 * x2**2 + 2 * x2 + 1
    y2.backward()

    print("\nFor y = 3*x^2 + 2*x + 1")
    print("x =", x2.item())
    print("y =", y2.item())
    print("dy/dx =", x2.grad.item())


def exercise_9():
    """Exercise 9: Learn polynomial coefficients using NAdam."""
    print("\n=== Exercise 9: Polynomial Coefficient Learning ===")

    # Generate data from y = x^2 + 2x + 3
    np.random.seed(42)
    x = np.linspace(-3, 3, 20)
    y = x**2 + 2 * x + 3

    xx = torch.tensor(
        np.column_stack((x**2, x, np.ones_like(x))),
        dtype=torch.float32,
    )
    yy = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

    w = torch.randn(3, 1, requires_grad=True)
    optimizer = torch.optim.NAdam([w], lr=0.01)

    for epoch in range(1000):
        optimizer.zero_grad()
        predictions = xx @ w
        loss = torch.mean((yy - predictions) ** 2)
        loss.backward()
        optimizer.step()

    print("Learned coefficients [a, b, c]:")
    print(w.detach().flatten())
    print("Final MSE:", loss.item())


def exercise_10():
    """Exercise 10: Solve a system of equations using optimization."""
    print("\n=== Exercise 10: Solving a System of Equations ===")

    torch.manual_seed(42)

    # Equations:
    # A + B = 9
    # C - D = 1
    # A + C = 8
    # B - D = 2
    variables = torch.randn(4, requires_grad=True)
    optimizer = torch.optim.NAdam([variables], lr=0.01)

    for epoch in range(2000):
        optimizer.zero_grad()

        A, B, C, D = variables
        loss = (
            (A + B - 9) ** 2
            + (C - D - 1) ** 2
            + (A + C - 8) ** 2
            + (B - D - 2) ** 2
        )

        loss.backward()
        optimizer.step()

    A, B, C, D = variables.detach()

    print(f"A = {A.item():.6f}")
    print(f"B = {B.item():.6f}")
    print(f"C = {C.item():.6f}")
    print(f"D = {D.item():.6f}")
    print("Final loss:", loss.item())

    print("\nCheck equations:")
    print("A + B =", (A + B).item(), "(target: 9)")
    print("C - D =", (C - D).item(), "(target: 1)")
    print("A + C =", (A + C).item(), "(target: 8)")
    print("B - D =", (B - D).item(), "(target: 2)")


if __name__ == "__main__":
    exercise_8()
    exercise_9()
    exercise_10()
