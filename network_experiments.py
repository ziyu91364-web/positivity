#!/usr/bin/env python3
"""Deterministic network positivity benchmarks for the theta method.

Run from the manuscript package with Python >=3.9, NumPy and SciPy:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 network_experiments.py

All step lengths use q_max=1.  The complete benchmark design is fixed below;
there is no graph search, parameter fitting, or random data selection.
"""
from __future__ import annotations

import csv
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import scipy
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.optimize import brentq


DESIGN = {
    "families": ["cycle", "square_torus", "heterogeneous_ring"],
    "sizes": [64, 256, 1024],
    "theta": [0.5, 0.75],
    "normalization": "All Q divided by the raw maximum exit rate: q_max=1.",
    "heterogeneous_ring": {
        "index": "i=0,...,n-1; x_i=2*pi*i/n",
        "edge_conductance": "c_i=exp(sin(x_i)+0.5*cos(3*x_i)); edge {i,i+1 mod n}",
        "vertex_mass": "m_i=exp(0.8*cos(x_i)+0.3*sin(5*x_i))",
        "generator": "Q_ij=c_ij/m_i off diagonal, Q_ii=-sum_{j!=i} Q_ij; pi_i=m_i/sum m",
    },
    "spectral_upper_bound": "Lambda=2*q_max, never the computed largest eigenvalue",
    "exact_reference": "Fourier eigenvalues for cycle and square_torus; full symmetric eigendecomposition for heterogeneous_ring; bracketed scalar root of min_i R_ii(theta*h)-(1-theta).",
    "validation": "Independent sparse LU solves of I+theta*h*A at h=(1 +/- 1e-6)*h_star and at H2; every diagonal checked. Certificate and reference ordering checked.",
    "root_relative_tolerance": 2e-14,
    "root_absolute_tolerance": 5e-14,
    "near_threshold_relative_offset": 1e-6,
}


def network(family: str, n: int):
    """Return symmetric A, pi, q and local second diagonal moment."""
    i = np.arange(n)
    masses = np.ones(n)
    if family in ("cycle", "heterogeneous_ring"):
        if family == "heterogeneous_ring":
            x = 2 * np.pi * i / n
            masses = np.exp(0.8 * np.cos(x) + 0.3 * np.sin(5 * x))
            conductance = np.exp(np.sin(x) + 0.5 * np.cos(3 * x))
        else:
            conductance = np.ones(n)
        source, target = i, (i + 1) % n
    elif family == "square_torus":
        side = int(np.sqrt(n))
        assert side * side == n
        source = np.concatenate((i, i))
        target = np.concatenate((side * (i // side) + (i + 1) % side,
                                 (i + side) % n))
        conductance = np.ones(2 * n)
    else:
        raise ValueError(family)
    rows = np.concatenate((source, target))
    cols = np.concatenate((target, source))
    data = np.concatenate((conductance, conductance))
    adjacency = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    raw_qmax = np.max(degree / masses)
    laplacian = sp.diags(degree) - adjacency
    inverse_sqrt_mass = sp.diags(1 / np.sqrt(masses))
    A = (inverse_sqrt_mass @ laplacian @ inverse_sqrt_mass / raw_qmax).tocsr()
    pi = masses / masses.sum()
    q = A.diagonal()
    second = np.asarray(A.multiply(A).sum(axis=1)).ravel()
    assert np.max(np.abs(A @ np.sqrt(pi))) < 2e-14
    assert abs(np.max(q) - 1) < 2e-14
    return A, pi, q, second, float(raw_qmax)


def certificates(pi, q, second, theta):
    """Evaluate the finite, mu<Lambda branches used by the fixed design."""
    w = 1 - pi
    mu = q / w
    nu = second / w
    Lambda = 2 * max(q)
    variance = nu - mu * mu
    assert min(variance) >= -1e-13
    delta = 1 - theta - pi
    # These explicit assertions distinguish this benchmark implementation from
    # a general-purpose routine for arbitrary theta, network sizes, or inputs.
    assert np.all(delta > 0) and np.all(mu < Lambda)
    H0 = 1 / ((1 - theta) * max(q))
    H1i = np.full(len(pi), np.inf)
    finite = delta > 0
    H1i[finite] = w[finite] / (q[finite] * delta[finite])
    a = (Lambda * mu - nu) / (Lambda - mu)
    assert min(a) > 0
    b = theta * (a + Lambda) - q
    discriminant = np.sqrt(b * b + 4 * theta * delta * a * Lambda)
    H2i = np.full(len(pi), np.inf)
    positive_b = finite & (b >= 0)
    negative_b = finite & (b < 0)
    H2i[positive_b] = ((b[positive_b] + discriminant[positive_b]) /
                      (2 * theta * delta[positive_b] * a[positive_b] * Lambda))
    H2i[negative_b] = 2 / (discriminant[negative_b] - b[negative_b])
    assert np.all(H2i >= H1i * (1 - 1e-13))
    return float(H0), float(min(H1i)), float(min(H2i))


def spectral_reference(family, A, pi):
    n = len(pi)
    if family == "cycle":
        eigenvalues = 1 - np.cos(2 * np.pi * np.arange(n) / n)
        return lambda s: np.array([np.mean(1 / (1 + s * eigenvalues))]), {
            "method": "analytic Fourier eigenvalues", "spectral_residual_max": 0.0,
            "lambda_max_reference_only": float(max(eigenvalues)),
        }
    if family == "square_torus":
        side = int(np.sqrt(n))
        cosine = np.cos(2 * np.pi * np.arange(side) / side)
        eigenvalues = (1 - 0.5 * (cosine[:, None] + cosine[None, :])).ravel()
        return lambda s: np.array([np.mean(1 / (1 + s * eigenvalues))]), {
            "method": "analytic Fourier eigenvalues", "spectral_residual_max": 0.0,
            "lambda_max_reference_only": float(max(eigenvalues)),
        }
    eigenvalues, eigenvectors = la.eigh(A.toarray(), check_finite=True,
                                      overwrite_a=True, driver="evd")
    residual = float(np.max(np.abs(A @ eigenvectors - eigenvectors * eigenvalues)))
    assert abs(eigenvalues[0]) < 2e-13 and eigenvalues[1] > 0
    stationary_error = float(min(np.max(np.abs(eigenvectors[:, 0] - np.sqrt(pi))),
                                 np.max(np.abs(eigenvectors[:, 0] + np.sqrt(pi)))))
    weights = eigenvectors[:, 1:] ** 2
    mass_error = float(max(np.abs(weights.sum(axis=1) - (1 - pi))))
    assert residual < 1e-12 and stationary_error < 1e-10 and mass_error < 1e-12
    return lambda s: pi + weights @ (1 / (1 + s * eigenvalues[1:])), {
        "method": "complete symmetric eigendecomposition with exact stationary atom",
        "spectral_residual_max": residual,
        "stationary_vector_error_max": stationary_error,
        "spectral_mass_error_max": mass_error,
        "lambda_max_reference_only": float(eigenvalues[-1]),
    }


def direct_diagonal(A, s):
    """Independent diagonal of the inverse from sparse LU, in small blocks."""
    n = A.shape[0]
    system = (sp.eye(n, format="csc") + s * A).tocsc()
    factor = sla.splu(system)
    diagonal = np.zeros(n)
    max_residual = 0.0
    for start in range(0, n, 64):
        stop = min(start + 64, n)
        indices = np.arange(start, stop)
        rhs = np.zeros((n, stop - start))
        rhs[indices, np.arange(stop - start)] = 1
        solution = factor.solve(rhs)
        diagonal[indices] = solution[indices, np.arange(stop - start)]
        max_residual = max(max_residual, float(np.max(np.abs(system @ solution - rhs))))
    return diagonal, max_residual


def run(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    validations = []
    for family in DESIGN["families"]:
        for n in DESIGN["sizes"]:
            A, pi, q, second, normalization = network(family, n)
            before = time.perf_counter()
            diagonal, spectral_checks = spectral_reference(family, A, pi)
            spectrum_seconds = time.perf_counter() - before
            for theta in DESIGN["theta"]:
                before = time.perf_counter()
                H0, H1, H2 = certificates(pi, q, second, theta)
                certificate_seconds = time.perf_counter() - before
                target = 1 - theta
                gap = lambda h: float(np.min(diagonal(theta * h)) - target)
                upper = H2 * 2
                while gap(upper) > 0:
                    upper *= 2
                before = time.perf_counter()
                exact = brentq(gap, 0, upper, xtol=5e-14, rtol=2e-14)
                root_seconds = time.perf_counter() - before
                assert H0 <= H1 * (1 + 1e-13) <= H2 * (1 + 2e-13)
                assert H2 <= exact * (1 + 1e-12)
                # Check every state using an independent factorization and solver.
                direct_checks = {}
                for name, h in (("H2", H2), ("below", exact * (1 - 1e-6)),
                                ("above", exact * (1 + 1e-6))):
                    direct, residual = direct_diagonal(A, theta * h)
                    spectral = diagonal(theta * h)
                    agreement = float(max(np.abs(direct - spectral)))
                    direct_checks[name] = {
                        "step": h,
                        "minimum_resolvent_gap": float(min(direct) - target),
                        "spectral_LU_diagonal_difference_max": agreement,
                        "linear_solve_residual_max": residual,
                    }
                    assert agreement < 2e-11 and residual < 2e-12
                assert direct_checks["H2"]["minimum_resolvent_gap"] >= -2e-12
                assert direct_checks["below"]["minimum_resolvent_gap"] > 1e-9
                assert direct_checks["above"]["minimum_resolvent_gap"] < -1e-9
                row = {
                    "family": family, "n": n, "theta": theta,
                    "H0": H0, "H1": H1, "H2": H2, "h_star": exact,
                    "H1_over_H0": H1 / H0, "H2_over_H0": H2 / H0,
                    "H2_over_H1": H2 / H1, "H2_over_h_star": H2 / exact,
                    "H0_over_h_star": H0 / exact, "H1_over_h_star": H1 / exact,
                    "pi_min": float(min(pi)), "pi_max": float(max(pi)),
                    "q_min": float(min(q)), "q_max": float(max(q)),
                    "nonzeros_Q": int(A.nnz), "raw_qmax": normalization,
                    "Lambda_used": 2 * float(max(q)),
                    "reference_spectrum_seconds": spectrum_seconds,
                    "certificate_computation_seconds": certificate_seconds,
                    "reference_root_seconds": root_seconds,
                }
                records.append(row)
                validations.append({"family": family, "n": n, "theta": theta,
                                    **spectral_checks, "direct_checks": direct_checks})
                print(f"{family:20s} n={n:4d} theta={theta:.2f} "
                      f"H0={H0:.9f} H1={H1:.9f} H2={H2:.9f} h*={exact:.9f} "
                      f"H2/H0={H2/H0:.4f} H2/h*={H2/exact:.4f}", flush=True)
    metadata = {
        "design": DESIGN,
        "environment": {"python": sys.version, "numpy": np.__version__,
                        "scipy": scipy.__version__, "platform": platform.platform(),
                        "thread_environment": {k: os.environ.get(k) for k in
                                               ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                                                "VECLIB_MAXIMUM_THREADS")}},
        "results": records, "validation": validations,
        "interpretation": "h_star is a floating-point numerical reference, not a certified interval or a symbolic exact value. Timings are single-run diagnostics, not performance comparisons.",
    }
    (output_dir / "network_results.json").write_text(json.dumps(metadata, indent=2) + "\n")
    with (output_dir / "network_results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    # Include the full design in this supplementary table, not selected successes.
    lines = [r"\begin{tabular}{llrrrrr}", r"Family & $n$ & $\theta$ & $H_0$ & $H_1$ & $H_2$ & $h_*$ \\", r"\hline"]
    labels = {"cycle": r"Cycle", "square_torus": r"Square torus", "heterogeneous_ring": r"Weighted ring"}
    for row in records:
        lines.append(f"{labels[row['family']]} & {row['n']} & {row['theta']:.2f} & "
                     f"{row['H0']:.4f} & {row['H1']:.4f} & {row['H2']:.4f} & "
                     f"{row['h_star']:.4f} " + r"\\")
    lines.append(r"\end{tabular}")
    (output_dir / "network_table_all.tex").write_text("\n".join(lines) + "\n")
    return metadata


if __name__ == "__main__":
    destination = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
    run(destination)
