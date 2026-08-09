# Matrix Linear Algebra Library v1.0

Python으로 구현한 행렬 연산 및 선형대수 알고리즘 라이브러리.

기본 행렬 연산, 선형시스템 풀이, 행렬 분해, 최소제곱법, 영공간, 거듭제곱법

---

## Features

### Basic Matrix Operations

- Matrix creation and validation
- Identity matrix
- Zero matrix
- Diagonal matrix
- Matrix addition
- Matrix subtraction
- Scalar multiplication
- Matrix multiplication
- Transpose
- Equality comparison
- Input data copy protection

### Matrix Properties

- Square matrix check
- Symmetric matrix check
- Zero matrix check
- Determinant
- Inverse matrix
- Rank
- Row Echelon Form (REF)
- Reduced Row Echelon Form (RREF)
- Frobenius norm
- Upper triangular check
- Lower triangular check
- Orthonormal column check
- Full column rank check

### Linear Systems

- Augmented matrix
- Analyze a linear system
  - Unique solution
  - No solution
  - Infinite solutions
- Check solution existence
- Solve a system with a unique solution
- Solve using LU decomposition
- Solve using PLU decomposition
- Null space basis
- General solution

For a system with solutions, the general solution is represented as

$$
x=x_p+c_1v_1+\cdots+c_kv_k
$$

where $x_p$ is a particular solution and $v_1,\dots,v_k$ are basis vectors of the null space.

### Least Squares

- Least squares solution using normal equations
- Least squares solution using QR decomposition
- Least squares prediction
- Residual vector
- Residual norm

The QR-based least squares solver uses

$$
A=QR,\qquad R\hat{x}=Q^Tb
$$

and solves the upper triangular system using back substitution.

### Vector Utilities

- Vector check
- Dot product
- Vector normalization
- Get a column vector
- Construct a matrix from column vectors

### Decomposition

- Gram–Schmidt process
- QR decomposition
- LU decomposition without row swaps
- PLU decomposition with partial pivoting
- Forward substitution
- Back substitution
- LU-based system solver
- PLU-based system solver

### Eigenvalue Approximation

- Dominant eigenvalue approximation
- Corresponding normalized eigenvector
- Power iteration
- Rayleigh quotient
- Residual-based convergence check

* The current power iteration implementation supports real symmetric matrices.

---

## Quick Start

### Basic matrix operations

```python
from Matrix import Matrix

A = Matrix([
    [1, 2],
    [3, 4]
])

B = Matrix([
    [5, 6],
    [7, 8]
])

print(A + B)
print(A - B)
print(A * 2)
print(A @ B)
print(A.transpose())
```

### Solve a linear system

```python
A = Matrix([
    [2, 1],
    [1, -1]
])

b = Matrix([
    [5],
    [1]
])

x = A.solve_unique(b)

print(x)
```

### General solution

```python
A = Matrix([
    [1, 2, 3],
    [2, 4, 6]
])

b = Matrix([
    [4],
    [8]
])

particular, basis = A.solve_general(b)

print("Particular solution:")
print(particular)

print("Null space basis:")
for vector in basis:
    print(vector)
```

The returned result represents

$$
x=x_p+c_1v_1+\cdots+c_kv_k
$$


where `particular` is $x_p$ and `basis` contains $v_1,\dots,v_k$.


### QR-based least squares

```python
A = Matrix([
    [1, 0],
    [0, 1],
    [1, 1]
])

b = Matrix([
    [1],
    [2],
    [2]
])

x_hat = A.least_squares_qr(b)

print(x_hat)
```

### Dominant eigenpair

```python
A = Matrix([
    [5, 0],
    [0, 2]
])

eigenvalue, eigenvector = A.dominant_eigenpair()

print("Dominant eigenvalue:", eigenvalue)
print("Eigenvector:")
print(eigenvector)
```

---

## Project Structure

```text
.
├── Matrix.py
├── README.md
└── tests
    ├── test_basic_operations.py
    ├── test_decompositions.py
    ├── test_eigenvalues.py
    ├── test_least_squares.py
    ├── test_linear_systems.py
    └── test_matrix_properties.py
```

---

## How to Test

Install pytest:

```bash
python3 -m pip install pytest
```

Check collected tests:

```bash
python3 -m pytest --collect-only -q
```

Run the complete test suite:

```bash
python3 -m pytest -v
```

Run a specific test file:

```bash
python3 -m pytest tests/test_eigenvalues.py -v
```

---

## Current Limitations

- The library currently supports real numbers only.
- LU decomposition does not perform row swaps. Use PLU decomposition when pivoting is required.
- QR decomposition requires linearly independent columns.
- QR-based least squares supports tall or square matrices with full column rank.
- Power iteration currently supports symmetric matrices.
- Power iteration may fail when the initial vector has no component in the dominant eigenvector direction.
- Numerical comparisons use an epsilon tolerance because of floating-point errors.

---

## Notes

This project is implemented for learning purposes.

The goal is not to replace libraries such as NumPy, but to understand how matrix operations and linear algebra algorithms work internally.

The implementation focuses on the mathematical principles behind Gaussian elimination, linear systems, matrix decomposition, least squares, null spaces, and eigenvalue approximation.