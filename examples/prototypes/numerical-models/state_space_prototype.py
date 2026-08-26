#!/usr/bin/env python3
"""State-Space Numerical Prototype Oracle.

Calculates continuous 2nd-order dynamical system simulation, Zero-Order Hold (ZOH)
discretization via matrix exponential Taylor series, discrete step simulation, and coordinate transformations.
Implemented in pure Python (standard library).
"""

import math


def mat_mul(A, B):
    r, m, c = len(A), len(A[0]), len(B[0])
    res = [[0.0] * c for _ in range(r)]
    for i in range(r):
        for k in range(m):
            for j in range(c):
                res[i][j] += A[i][k] * B[k][j]
    return res


def mat_add(A, B):
    return [
        [A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))
    ]


def mat_scale(A, s):
    return [[A[i][j] * s for j in range(len(A[0]))] for i in range(len(A))]


def mat_exp_taylor(M, order=20):
    """Compute matrix exponential via Taylor series."""
    n = len(M)
    res = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    term = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    for k in range(1, order + 1):
        term = mat_mul(term, M)
        term = mat_scale(term, 1.0 / float(k))
        res = mat_add(res, term)

    return res


def print_mat(name, M):
    print(f"{name}:")
    for row in M:
        print("  [" + ", ".join(f"{val:12.6f}" for val in row) + "]")


def main():
    print("=== State-Space Numerical Prototype Oracle ===")

    # 1. 2nd-Order Continuous Spring-Mass-Damper System
    # \ddot{x} + 0.8 \dot{x} + 4 x = u
    # State: x_1 = pos, x_2 = vel
    A_c = [[0.0, 1.0], [-4.0, -0.8]]
    B_c = [[0.0], [1.0]]
    C_c = [[1.0, 0.0]]
    D_c = [[0.0]]

    print("\n--- Continuous-Time System ---")
    print_mat("A_c", A_c)
    print_mat("B_c", B_c)
    print_mat("C_c", C_c)
    print_mat("D_c", D_c)

    # Derivative at state x = [1.0, 0.5]^T with input u = 0.0
    x_test = [[1.0], [0.5]]
    u_test = [[0.0]]
    Ax = mat_mul(A_c, x_test)
    Bu = mat_mul(B_c, u_test)
    x_dot = mat_add(Ax, Bu)
    y_test = mat_add(mat_mul(C_c, x_test), mat_mul(D_c, u_test))

    print_mat("x_dot at [1.0, 0.5]^T", x_dot)
    print(f"y at [1.0, 0.5]^T: {y_test[0][0]:.6f}")

    # 2. Exact ZOH Discretization for Ts = 0.05s
    # Augmented matrix M = [[A, B], [0, 0]] * dt of size 3x3
    dt = 0.05
    M = [
        [A_c[0][0] * dt, A_c[0][1] * dt, B_c[0][0] * dt],
        [A_c[1][0] * dt, A_c[1][1] * dt, B_c[1][0] * dt],
        [0.0, 0.0, 0.0],
    ]
    exp_M = mat_exp_taylor(M, order=20)

    A_d = [[exp_M[0][0], exp_M[0][1]], [exp_M[1][0], exp_M[1][1]]]
    B_d = [[exp_M[0][2]], [exp_M[1][2]]]
    C_d = [row[:] for row in C_c]
    D_d = [row[:] for row in D_c]

    print(f"\n--- Discrete-Time System (ZOH, Ts = {dt}s) ---")
    print_mat("A_d", A_d)
    print_mat("B_d", B_d)
    print_mat("C_d", C_d)
    print_mat("D_d", D_d)

    # 3. 20-step Discrete Unit Step Simulation (u[k] = 1.0, x[0] = 0)
    num_steps = 20
    x_k = [[0.0], [0.0]]
    u_step = [[1.0]]

    print("\n--- 20-Step Unit Step Trajectory ---")
    print(f"{'Step':<6}{'x_1 (pos)':<16}{'x_2 (vel)':<16}{'y (output)':<16}")
    for k in range(num_steps):
        y_k = mat_add(mat_mul(C_d, x_k), mat_mul(D_d, u_step))
        print(f"{k:<6}{x_k[0][0]:<16.8f}{x_k[1][0]:<16.8f}{y_k[0][0]:<16.8f}")
        x_next = mat_add(mat_mul(A_d, x_k), mat_mul(B_d, u_step))
        x_k = x_next

    # 4. Similarity Coordinate Transformation: T = [[1, 1], [0, 1]], T^-1 = [[1, -1], [0, 1]]
    T = [[1.0, 1.0], [0.0, 1.0]]
    T_inv = [[1.0, -1.0], [0.0, 1.0]]

    # A_trans = T * A * T^-1
    A_trans = mat_mul(mat_mul(T, A_d), T_inv)
    B_trans = mat_mul(T, B_d)
    C_trans = mat_mul(C_d, T_inv)

    print("\n--- Transformed System (T = [[1, 1], [0, 1]]) ---")
    print_mat("A_tilde", A_trans)
    print_mat("B_tilde", B_trans)
    print_mat("C_tilde", C_trans)


if __name__ == "__main__":
    main()
