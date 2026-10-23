
import torch


def exercise_1():
    """Create an integer tensor and inspect its properties."""
    a = torch.tensor(
        [[1, 2, 3],
         [4, 5, 6]],
        dtype=torch.int32
    )

    print("=== Exercise 1: Integer Tensor ===")
    print(a)
    print("Shape:", a.shape)
    print("Dimensions:", a.ndim)
    print("Data type:", a.dtype)


def exercise_2():
    """Create commonly used tensors."""
    torch.manual_seed(42)

    a = torch.rand(3, 4)
    b = torch.randn(3, 4)
    c = torch.zeros(2, 3)
    d = torch.ones(2, 3)
    e = torch.eye(4)
    f = torch.full((5, 5), 10)

    print("\n=== Exercise 2: Tensor Initialization ===")
    print("rand:\n", a)
    print("randn:\n", b)
    print("zeros:\n", c)
    print("ones:\n", d)
    print("eye:\n", e)
    print("full:\n", f)


def exercise_3():
    """Inspect a three-dimensional tensor."""
    a = torch.zeros(2, 3, 4)

    print("\n=== Exercise 3: Tensor Properties ===")
    print(a)
    print("Shape:", a.shape)
    print("Size:", a.size())
    print("Dimensions:", a.ndim)
    print("Data type:", a.dtype)


def exercise_4():
    """Practice tensor indexing and slicing."""
    torch.manual_seed(42)
    a = torch.randn(3, 4, 5)

    print("\n=== Exercise 4: Indexing and Slicing ===")
    print("Original tensor:\n", a)
    print("a[1]:\n", a[1])
    print("a[1:, 2:4]:\n", a[1:, 2:4])
    print("Last two elements along the third dimension:\n",
          a[:, :, -2:])
    print("Shape of a[1]:", a[1].shape)
    print("Shape of a[1:, 2:4]:", a[1:, 2:4].shape)


def exercise_5():
    """Practice reshaping and flattening."""
    torch.manual_seed(42)
    a = torch.randn(3, 4)

    print("\n=== Exercise 5: Reshape and Flatten ===")
    print("Original tensor:\n", a)
    print("Flattened tensor:\n", a.ravel())
    print("Reshape to (3, 2, 2):\n", a.reshape(3, 2, 2))
    print("Reshape to (2, 6):\n", a.reshape(2, 6))
    print("Number of elements:", a.numel())


def exercise_6():
    """Perform element-wise and matrix operations."""
    torch.manual_seed(42)
    a = torch.randn(2, 3)
    b = torch.randn(2, 3)

    print("\n=== Exercise 6: Tensor Operations ===")
    print("A:\n", a)
    print("B:\n", b)
    print("A + B:\n", a + b)
    print("A / B:\n", a / b)
    print("A squared:\n", a ** 2)
    print("A @ B.T:\n", a @ b.T)


def exercise_7():
    """Calculate statistical measures."""
    torch.manual_seed(42)
    a = torch.randn(3, 4)

    print("\n=== Exercise 7: Statistics ===")
    print("Tensor:\n", a)
    print("Mean:", a.mean())
    print("Standard deviation:", a.std())
    print("Cumulative sum along dimension 0:\n", a.cumsum(dim=0))
    print("Cumulative sum along dimension 1:\n", a.cumsum(dim=1))


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
    exercise_5()
    exercise_6()
    exercise_7()
