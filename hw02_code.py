
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

from __future__ import annotations

import argparse
import csv
import math
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


