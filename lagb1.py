#!/usr/bin/env python3
import numpy as np

# 1. Vectors and Matrices
# Create a 3x3 matrix
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

# Create a 3-dimensional vector
v = np.array([1, 2, 3])

# 2. Matrix-Vector Multiplication
result = np.dot(A, v)
print("Matrix-Vector Multiplication (A * v):")
print(result)

# 3. Dot Product of Two Vectors
# Create another vector
b = np.array([4, 5, 6])
dot_product = np.dot(v, b)
print("\nDot Product of v and b:")
print(dot_product)

# 4. Matrix Transpose
transpose_A = np.transpose(A)
print("\nTranspose of Matrix A:")
print(transpose_A)

# 5. Eigenvalues and Eigenvectors
# Calculate eigenvalues and eigenvectors of the matrix A
eigenvalues, eigenvectors = np.linalg.eig(A)
print("\nEigenvalues of Matrix A:")
print(eigenvalues)

print("\nEigenvectors of Matrix A:")
print(eigenvectors)