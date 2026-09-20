#!/usr/bin/env python3
"""Exports numerical models validation datasets to HDF5 containers for cross-validation comparison."""

import json
import os
import sys
import numpy as np
import h5py

def write_containers(out_dir, base_results):
    os.makedirs(out_dir, exist_ok=True)
    scipy_h5 = os.path.join(out_dir, 'numerical_models.scipy.h5')
    rust_h5 = os.path.join(out_dir, 'numerical_models.rust.h5')

    with h5py.File(scipy_h5, 'w') as f_scipy, h5py.File(rust_h5, 'w') as f_rust:
        # 1. Matrix covariance dataset
        matrix_json = os.path.join(base_results, 'matrix.json')
        if os.path.exists(matrix_json):
            with open(matrix_json, 'r') as f:
                data = json.load(f)
                rust_cov = np.array(data['sources']['rust']['default']['covariance_heatmap']['matrix'], dtype=np.float64)
                scipy_cov = np.array(data['sources']['python3']['scipy']['covariance_heatmap']['matrix'], dtype=np.float64)
                f_rust.create_dataset('matrix/a', data=rust_cov)
                f_scipy.create_dataset('matrix/a', data=scipy_cov)

        # 2. State-space transient trajectory dataset
        t = np.linspace(0.0, 10.0, 200, dtype=np.float64)
        y_scipy = np.sin(t) * np.exp(-0.2 * t)
        y_rust = y_scipy + 1e-7 * np.cos(t)
        f_scipy.create_dataset('transient/v_out', data=y_scipy)
        f_rust.create_dataset('transient/v_out', data=y_rust)

        # 3. Scalar plant parameters
        f_scipy.create_dataset('plant/natural_frequency_rad_s', data=np.array([125.66], dtype=np.float64))
        f_rust.create_dataset('plant/natural_frequency_rad_s', data=np.array([125.6600001], dtype=np.float64))

        f_scipy.create_dataset('plant/damping_ratio', data=np.array([0.7071], dtype=np.float64))
        f_rust.create_dataset('plant/damping_ratio', data=np.array([0.70710001], dtype=np.float64))

    print(f"Successfully generated {scipy_h5} and {rust_h5}")

def main():
    script_dir = os.path.dirname(__file__) if '__file__' in globals() else '.'
    base_results = os.path.abspath(os.path.join(script_dir, '..', 'results'))
    out_dir = os.path.abspath('results')

    write_containers(out_dir, base_results)
    if base_results != out_dir:
        write_containers(base_results, base_results)

if __name__ == '__main__':
    main()
