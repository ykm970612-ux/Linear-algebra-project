# Matrix Linear Algebra Library

Python으로 직접 구현한 행렬 연산 및 선형대수 알고리즘 라이브러리.

기본 행렬 연산, RREF, 선형시스템 풀이, Least Squares, Gram-Schmidt, QR Decomposition.

---

## Features

### Basic Matrix Operations

- Matrix addition
- Matrix subtraction
- Scalar multiplication
- Matrix multiplication
- Transpose
- Equality comparison

### Matrix Properties

- Determinant
- Inverse matrix
- Rank
- Row Echelon Form
- Reduced Row Echelon Form
- Frobenius norm
- Upper triangular check
- Orthonormal check
- Full column rank check

### Linear Systems

- Augmented matrix
- Analyze system
  - unique solution
  - no solution
  - infinite solutions
- Check solution existence
- Solve system with a unique solution

### Least Squares

- Least squares solution
- Least squares prediction
- Residual vector
- Residual norm

### Vector Utilities

- Vector check
- Dot product
- Normalize
- Get column vector
- Construct matrix from column vectors

### Decomposition

- Gram-Schmidt process
- QR decomposition

---

## Project Structure

```text
.
├── Matrix.py
├── test_matrix.py
└── README.md
```

## How to tests

```bash
python3 test_matrix.py

```

---

## Notes

This project is implemented for learning purposes.

The goal is not to replace libraries such as NumPy, but to understand how matrix operations and linear algebra algorithms work internally.

