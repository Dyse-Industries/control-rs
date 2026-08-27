#!/usr/bin/env python3
"""Matrix Numerical Prototype Oracle.

Calculates reference matrix linear solver, LU decomposition, matrix inversion,
transposition, and matrix arithmetic.
Implemented in pure Python (standard library) with optional NumPy verification.
"""

import math


def mat_mul(A, B):
    """Multiply matrix A (r x m) by B (m x c)."""
    r, m, c = len(A), len(A[0]), len(B[0])
    res = [[0.0] * c for _ in range(r)]
    for i in range(r):
        for k in range(m):
            for j in range(c):
                res[i][j] += A[i][k] * B[k][j]
    return res


def mat_add(A, B):
    """Add matrix A and B."""
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def mat_sub(A, B):
    """Subtract matrix B from A."""
    return [[A[i][j] - B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def mat_transpose(A):
    """Transpose matrix A."""
    r, c = len(A), len(A[0])
    return [[A[i][j] for i in range(r)] for j in range(c)]


def lu_decompose(A):
    """Perform Doolittle LU decomposition with partial pivoting."""
    n = len(A)
    L = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    U = [[float(A[i][j]) for j in range(n)] for i in range(n)]
    piv = list(range(n))

    for k in range(n):
        # Pivot selection
        max_row = k
        max_val = abs(U[k][k])
        for i in range(k + 1, n):
            if abs(U[i][k]) > max_val:
                max_val = abs(U[i][k])
                max_row = i

        if max_row != k:
            U[k], U[max_row] = U[max_row], U[k]
            piv[k], piv[max_row] = piv[max_row], piv[k]
            for j in range(k):
                L[k][j], L[max_row][j] = L[max_row][j], L[k][j]

        pivot = U[k][k]
        if abs(pivot) < 1e-14:
            raise ValueError("Singular matrix")

        for i in range(k + 1, n):
            factor = U[i][k] / pivot
            L[i][k] = factor
            for j in range(k, n):
                U[i][j] -= factor * U[k][j]

    return L, U, piv


def lu_solve(L, U, piv, b):
    """Solve Ax = b using LU factors and pivot permutation."""
    n = len(L)
    # Apply permutation to b: Pb
    pb = [b[piv[i]][0] for i in range(n)]

    # Forward substitution: Ly = pb
    y = [0.0] * n
    for i in range(n):
        s = pb[i]
        for j in range(i):
            s -= L[i][j] * y[j]
        y[i] = s / L[i][i]

    # Backward substitution: Ux = y
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = y[i]
        for j in range(i + 1, n):
            s -= U[i][j] * x[j]
        x[i] = s / U[i][i]

    return [[val] for val in x]


def mat_inv(A):
    """Invert matrix A via column-by-column LU solve."""
    n = len(A)
    L, U, piv = lu_decompose(A)
    inv = [[0.0] * n for _ in range(n)]
    for j in range(n):
        e_j = [[1.0 if i == j else 0.0] for i in range(n)]
        col_j = lu_solve(L, U, piv, e_j)
        for i in range(n):
            inv[i][j] = col_j[i][0]
    return inv


def print_mat(name, M):
    print(f"{name}:")
    for row in M:
        print("  [" + ", ".join(f"{val:12.6f}" for val in row) + "]")


def main():
    print("=== Matrix Numerical Prototype Oracle ===")

    # 1. 2x2 Matrix Arithmetic
    M1 = [[1.0, 2.0], [3.0, 4.0]]
    M2 = [[5.0, 6.0], [7.0, 8.0]]

    print("\n--- Matrix Construction & Basic Arithmetic ---")
    print_mat("M1", M1)
    print_mat("M2", M2)

    print_mat("M1 + M2", mat_add(M1, M2))
    print_mat("M2 - M1", mat_sub(M2, M1))
    print_mat("M1 * M2", mat_mul(M1, M2))
    print_mat("M1^T (Transpose)", mat_transpose(M1))

    # 2. 3x3 Linear System Solve: A * x = b
    A = [[3.0, 2.0, -1.0], [2.0, -2.0, 4.0], [-1.0, 0.5, -1.0]]
    b = [[1.0], [-2.0], [0.0]]

    print("\n--- Linear System A * x = b ---")
    print_mat("A", A)
    print_mat("b", b)

    L, U, piv = lu_decompose(A)
    print_mat("L", L)
    print_mat("U", U)
    print(f"Pivots: {piv}")

    x = lu_solve(L, U, piv, b)
    print_mat("Solution x", x)

    # Residual check: ||A * x - b||
    Ax = mat_mul(A, x)
    res = math.sqrt(sum((Ax[i][0] - b[i][0]) ** 2 for i in range(len(b))))
    print(f"Residual norm: {res:.16e}")

    # 3. Matrix Inversion & Identity Check
    A_inv = mat_inv(A)
    print("\n--- Matrix Inversion & Identity Check ---")
    print_mat("A^-1", A_inv)

    ident_check = mat_mul(A, A_inv)
    print_mat("A * A^-1 (Identity check)", ident_check)


if __name__ == "__main__":
    main()
