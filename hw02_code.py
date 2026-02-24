
"""
Lily Erickson 
ASTR 5900 HW2: Root Finding (Bisection and Newton-Raphson)

This script contains all of the code for Problem #2 (parts a,b,c) and Problem #3
  Problem 2(a): Solve f(x)=0 using bisection + Newton-Raphson with epsilon < 1e-8.
               Print step-by-step outputs and report total iterations.
  Problem 2(b): Test sensitivity to initial guesses/brackets (3–4 cases) and compare iterations.
  Problem 2(c): Newton-Raphson: test initial guesses that can lead to failure or unstable behavior.
  Problem 3:    Solve an additional physics/astro equation with no simple closed-form solution
               (Kepler's equation) for multiple parameter values.



Notes to self: 
  - use python3 to run code on terminal
"""

import argparse
import csv
import math
import os
import matplotlib.pyplot as plt
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

# ---------------------------
# Problem 1 function
# ---------------------------
def f(x: float) -> float:
    """f(x) = x^3 - 7x^2 + 14x - 5"""
    return x**3 - 7*x**2 + 14*x - 5


def fp(x: float) -> float:
    """f'(x) = 3x^2 - 14x + 14"""
    return 3*x**2 - 14*x + 14


# ---------------------------
# Utilities
# ---------------------------
def rel_err(new: float, old: float) -> float:
    """Relative error used in the lectures: |(new-old)/new|. Falls back if new==0."""
    if new == 0.0:
        return abs(new - old)
    return abs((new - old) / new)


def ensure_dirs(*paths: str) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)


def write_csv(path: str, header: List[str], rows: List[List[float]]) -> None:
    with open(path, "w", encoding="utf-8") as fcsv:
        fcsv.write(",".join(header) + "\n")
        for r in rows:
            fcsv.write(",".join(str(v) for v in r) + "\n")


def write_latex_table(path: str, caption: str, label: str, header: List[str], rows: List[List[float]]) -> None:
    """
    Writes a LaTeX table using longtable (good for many rows).
    """
    colspec = " | ".join(["c"] * len(header))
    with open(path, "w", encoding="utf-8") as ftex:
        ftex.write("\\begin{longtable}{" + colspec + "}\n")
        ftex.write(f"\\caption{{{caption}}}\\\\\n")
        ftex.write(f"\\label{{{label}}}\\\\\n")
        ftex.write("\\hline\n")
        ftex.write(" & ".join(header) + " \\\\\n")
        ftex.write("\\hline\n")
        ftex.write("\\endfirsthead\n")
        ftex.write("\\hline\n")
        ftex.write(" & ".join(header) + " \\\\\n")
        ftex.write("\\hline\n")
        ftex.write("\\endhead\n")
        for r in rows:
            # formats numbers nicely for LaTeX
            formatted = []
            for v in r:
                if isinstance(v, float):
                    formatted.append(f"{v:.12g}")
                else:
                    formatted.append(str(v))
            ftex.write(" & ".join(formatted) + " \\\\\n")
        ftex.write("\\hline\n")
        ftex.write("\\end{longtable}\n")


@dataclass
class RootResult:
    root: float
    iterations: int
    history: List[List[float]]  # numeric rows for saving/plotting


# ---------------------------
# Bisection
# ---------------------------
def bisection(
    func: Callable[[float], float],
    a: float,
    b: float,
    eps: float = 1e-8,
    max_iter: int = 10_000,
    verbose: bool = True,
) -> RootResult:
    """
    Bisection method using relative error on successive midpoints:
      epsilon_n = |(x_{n} - x_{n-1}) / x_{n}|
    Stops when epsilon_n < eps.

    Requires f(a) and f(b) have opposite signs.
    """
    fa = func(a)
    fb = func(b)

    if fa == 0.0:
        return RootResult(root=a, iterations=0, history=[])
    if fb == 0.0:
        return RootResult(root=b, iterations=0, history=[])

    if fa * fb > 0.0:
        raise ValueError("Bisection requires a bracket: f(a) and f(b) must have opposite signs.")

    history: List[List[float]] = []
    x_old: Optional[float] = None

    if verbose:
        print("\n=== Bisection Method ===")
        print(f"Start: a={a}, b={b}, f(a)={fa}, f(b)={fb}, eps={eps}")
        print("iter, a, b, m, f(m), rel_err")

    for i in range(1, max_iter + 1):
        m = 0.5 * (a + b)
        fm = func(m)

        rerr = float("nan") if x_old is None else rel_err(m, x_old)

        history.append([i, a, b, m, fm, rerr])

        if verbose:
            print(f"{i:4d}, {a:.12g}, {b:.12g}, {m:.12g}, {fm:.12g}, {rerr:.12g}")

        if x_old is not None and rerr < eps:
            return RootResult(root=m, iterations=i, history=history)

        # Update bracket
        if fa * fm < 0.0:
            b = m
            fb = fm
        else:
            a = m
            fa = fm

        x_old = m

    return RootResult(root=m, iterations=max_iter, history=history)


# ---------------------------
# Newton-Raphson
# ---------------------------
def newton(
    func: Callable[[float], float],
    dfunc: Callable[[float], float],
    x0: float,
    eps: float = 1e-8,
    max_iter: int = 10_000,
    verbose: bool = True,
    deriv_floor: float = 0.0,
) -> RootResult:
    """
    Newton-Raphson using relative error based on the Newton step:
      dx = f(x)/f'(x)
      x_{n+1} = x_n - dx
      epsilon_n = |dx / x_{n+1}|
    Stops when epsilon_n < eps.

    deriv_floor: if >0, we treat |f'(x)| < deriv_floor as failure (prevents crazy blow-ups).
    """
    history: List[List[float]] = []
    x = x0

    if verbose:
        print("\n=== Newton-Raphson Method ===")
        print(f"Start: x0={x0}, eps={eps}")
        print("iter, x_n, f(x_n), f'(x_n), dx, rel_err")

    for i in range(1, max_iter + 1):
        fx = func(x)
        fpx = dfunc(x)

        if deriv_floor > 0.0 and abs(fpx) < deriv_floor:
            raise ZeroDivisionError(f"Derivative too small: f'(x)={fpx} at x={x}")

        if fpx == 0.0:
            raise ZeroDivisionError(f"Derivative is zero at x={x}; Newton fails here.")

        dx = fx / fpx
        x_new = x - dx
        rerr = abs(dx / x_new) if x_new != 0.0 else abs(dx)

        history.append([i, x, fx, fpx, dx, rerr])

        if verbose:
            print(f"{i:4d}, {x:.12g}, {fx:.12g}, {fpx:.12g}, {dx:.12g}, {rerr:.12g}")

        if rerr < eps:
            return RootResult(root=x_new, iterations=i, history=history)

        x = x_new

    return RootResult(root=x, iterations=max_iter, history=history)

# ---------------------------
# Plotting
# ---------------------------
def plot_error(history: List[List[float]], method_name: str, outpath: str) -> None:
    """
    Plots relative error vs iteration on a log scale.
    history format:
      Bisection rows: [iter, a, b, m, f(m), rel_err]
      Newton rows:    [iter, x, f(x), f'(x), dx, rel_err]
    """
    iters = [int(r[0]) for r in history]
    errs = [float(r[-1]) for r in history]

    # remove nan (first step has no previous value for bisection)
    iters2 = []
    errs2 = []
    for it, e in zip(iters, errs):
        if not (isinstance(e, float) and math.isnan(e)) and e > 0:
            iters2.append(it)
            errs2.append(e)

    plt.figure()
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Relative error")
    plt.title(f"{method_name}: Relative error vs iteration")
    plt.plot(iters2, errs2, marker="o", linestyle="-")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


# ---------------------------------------------
# Experiments for Problem 2 (b) & (c)
# ---------------------------------------------
def sensitivity_experiments(eps: float = 1e-8) -> None:
    """
    Runs a few different initial guesses/brackets and prints iteration counts.
    Also demonstrates Newton going "astray" if you pick derivative-zero points.
    """
    print("\n\n=== Q2(b) Sensitivity experiments (iteration counts) ===")

    # Bisection sensitivity (different brackets that still contain the root)
    bisect_brackets = [
        (0.0, 1.0),
        (0.3, 0.6),
        (0.4, 0.5),
        (0.45, 0.46),
        (-1.0, 1.0),
    ]

    print("\nBisection brackets:")
    for a, b in bisect_brackets:
        try:
            res = bisection(f, a, b, eps=eps, verbose=False)
            print(f"  bracket [{a}, {b}] -> root ~ {res.root:.12g}, iterations = {res.iterations}")
        except Exception as e:
            print(f"  bracket [{a}, {b}] -> FAILED: {e}")

    # Newton sensitivity (different x0)
    newton_guesses = [0.0, 0.5, 1.0, -1.0, 2.0, 5.0, 10.0]
    print("\nNewton initial guesses:")
    for x0 in newton_guesses:
        try:
            res = newton(f, fp, x0, eps=eps, verbose=False)
            print(f"  x0={x0:>7} -> root ~ {res.root:.12g}, iterations = {res.iterations}")
        except Exception as e:
            print(f"  x0={x0:>7} -> FAILED: {e}")

    print("\n\n=== Q2(c) Newton 'astray' examples ===")
    print("Newton can fail if f'(x0)=0 (division by zero) or behave poorly if |f'(x0)| is tiny.")

    # For this problem: f'(x)=3x^2-14x+14 has zeros near x≈1.451416 and x≈3.215250
    bad_points = [1.45141623, 3.21525044]  # approx locations where derivative is near 0
    for x0 in bad_points:
        try:
            res = newton(f, fp, x0, eps=eps, verbose=False, deriv_floor=1e-12)
            print(f"  x0≈{x0:.8f} -> root ~ {res.root:.12g}, iterations = {res.iterations} (often slow/unstable)")
        except Exception as e:
            print(f"  x0≈{x0:.8f} -> FAILED/ASTRAY: {e}")

    print("\nConceptual note: if you chose x0 EXACTLY where f'(x0)=0, Newton fails immediately.")


# ---------------------------
# Problem 3: Kepler's Equation 
#   Solve:  M = E - e sin(E)
#   Root form: g(E) = E - e sin(E) - M = 0
#   No closed-form solution for E in general (when e != 0)
# ---------------------------

def kepler_g(E: float, e: float, M: float) -> float:
    """g(E) = E - e sin(E) - M"""
    return E - e * math.sin(E) - M

def kepler_gp(E: float, e: float, M: float) -> float:
    """g'(E) = 1 - e cos(E)"""
    return 1.0 - e * math.cos(E)

def run_problem3(eps: float, max_iter: int, verbose: bool, no_save: bool) -> None:
    """
    Runs Kepler equation solves for a few (e, M) cases using both methods.
    Saves a results summary table + a figure of iteration count vs eccentricity.
    """

    print("\n\n=== Problem 3: Kepler's Equation ===")
    print("Solve g(E)=E - e sin(E) - M = 0 for E (eccentric anomaly).")

    # Keeping M fixed while increasing e shows harder nonlinear behavior.
    cases = [
        (0.2, 1.0),
        (0.6, 1.0),
        (0.9, 1.0),
    ]

    # For e < 1, g'(E) = 1 - e cos(E) >= 1 - e > 0 so g(E) is monotonic.
    a = 0.0
    b = 2.0 * math.pi

    results_rows = []  # e, M, method, root, iterations, residual

    # For plotting iterations vs eccentricity
    es = []
    it_bis = []
    it_new = []

    for (e, M) in cases:
        # closure functions for this case
        g = lambda E, ee=e, MM=M: kepler_g(E, ee, MM)
        gp = lambda E, ee=e, MM=M: kepler_gp(E, ee, MM)

        # Bisection
        bres = bisection(g, a, b, eps=eps, max_iter=max_iter, verbose=verbose)
        b_resid = abs(g(bres.root))
        results_rows.append([e, M, "bisection", bres.root, bres.iterations, b_resid])

        # Newton: good standard initial guess is E0 = M
        nres = newton(g, gp, x0=M, eps=eps, max_iter=max_iter, verbose=verbose)
        n_resid = abs(g(nres.root))
        results_rows.append([e, M, "newton", nres.root, nres.iterations, n_resid])

        es.append(e)
        it_bis.append(bres.iterations)
        it_new.append(nres.iterations)

        print(f"\nCase e={e:.2f}, M={M:.2f}:")
        print(f"  Bisection: E ~ {bres.root:.12g}, iters={bres.iterations}, |g(E)|~{b_resid:.3e}")
        print(f"  Newton:    E ~ {nres.root:.12g}, iters={nres.iterations}, |g(E)|~{n_resid:.3e}")

    # Save summary outputs
    if not no_save:
        ensure_dirs("figures", "tables")

        header = ["e", "M", "method", "E_root", "iterations", "|g(E_root)|"]
        write_csv("tables/problem3_kepler_results.csv", header, results_rows)

        write_latex_table(
            "tables/problem3_kepler_results.tex",
            caption="Problem 3 results for Kepler's equation $M = E - e\\sin E$ solved via bisection and Newton-Raphson.",
            label="tab:kepler_results",
            header=header,
            rows=results_rows,
        )

        # Figure: iteration count vs eccentricity (shows Newton sensitivity / nonlinearity)
        plt.figure()
        plt.xlabel("Eccentricity e")
        plt.ylabel("Iterations")
        plt.title("Kepler Equation: Iterations vs Eccentricity")
        plt.plot(es, it_bis, marker="o", linestyle="-", label="Bisection")
        plt.plot(es, it_new, marker="o", linestyle="-", label="Newton")
        plt.legend()
        plt.tight_layout()
        plt.savefig("figures/problem3_kepler_iterations.png", dpi=200)
        plt.close()

        print("\nSaved Problem 3 table to tables/ and figure to figures/ ✅")


# ---------------------------
# Main (Problem 2a is default behavior)
# ---------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="HW02 Root Finding: Bisection + Newton")
    parser.add_argument("--eps", type=float, default=1e-8, help="Relative error tolerance epsilon")
    parser.add_argument("--a", type=float, default=0.0, help="Bisection left bracket a")
    parser.add_argument("--b", type=float, default=1.0, help="Bisection right bracket b")
    parser.add_argument("--x0", type=float, default=0.0, help="Newton initial guess x0")
    parser.add_argument("--maxiter", type=int, default=10000, help="Max iterations")
    parser.add_argument("--no-save", action="store_true", help="Do not save tables/plots")
    parser.add_argument("--no-verbose", action="store_true", help="Do not print each step")
    parser.add_argument("--run-experiments", action="store_true", help="Run Q2(b)(c) experiments")
    parser.add_argument("--problem3", action="store_true", help="Run Problem 3 (Kepler equation)")
    args = parser.parse_args()

    eps = args.eps
    verbose = not args.no_verbose

    ensure_dirs("figures", "tables")

    # --- Problem #2(a): runs both methods on Problem 1's equation with eps < 1e-8 and shows outputs each step ---
    bres = bisection(f, args.a, args.b, eps=eps, max_iter=args.maxiter, verbose=verbose)
    nres = newton(f, fp, args.x0, eps=eps, max_iter=args.maxiter, verbose=verbose)

    print("\n\n=== Summary (Q2a) ===")
    print(f"Bisection: root ~ {bres.root:.12g}, iterations = {bres.iterations}, f(root) ~ {f(bres.root):.3e}")
    print(f"Newton:    root ~ {nres.root:.12g}, iterations = {nres.iterations}, f(root) ~ {f(nres.root):.3e}")

    if not args.no_save:
        # Save bisection tables
        b_header = ["iter", "a", "b", "m", "f(m)", "rel_err"]
        write_csv("tables/bisection_steps.csv", b_header, bres.history)
        write_latex_table(
            "tables/bisection_steps.tex",
            caption="Bisection method iteration outputs for $f(x)=x^3-7x^2+14x-5$.",
            label="tab:bisection_steps",
            header=b_header,
            rows=bres.history,
        )

        # Save newton tables
        n_header = ["iter", "x_n", "f(x_n)", "f'(x_n)", "dx", "rel_err"]
        write_csv("tables/newton_steps.csv", n_header, nres.history)
        write_latex_table(
            "tables/newton_steps.tex",
            caption="Newton-Raphson iteration outputs for $f(x)=x^3-7x^2+14x-5$.",
            label="tab:newton_steps",
            header=n_header,
            rows=nres.history,
        )

        # Plots
        plot_error(bres.history, "Bisection", "figures/bisection_error.png")
        plot_error(nres.history, "Newton-Raphson", "figures/newton_error.png")
        print("\nSaved tables to ./tables and figures to ./figures ✅")

    # --- Problem #2 (b) & (c): sensitivity / astray experiments ---
    if args.run_experiments:
        sensitivity_experiments(eps=eps)

    # --- Problem #3: Kepler equation application ---
    if args.problem3:
        run_problem3(eps=eps, max_iter=args.maxiter, verbose=verbose, no_save=args.no_save)


if __name__ == "__main__":
    main()
