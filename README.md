# Deep Residual Learning for Volterra Dynamic Integral Equations on Arbitrary Time Scales

Official implementation accompanying the paper

> **Deep Residual Learning for Volterra Dynamic Integral Equations on Arbitrary Time Scales**

---

## Overview

This repository presents a neural residual framework for solving nonlinear Volterra dynamic integral equations on arbitrary time scales. The proposed methodology combines time-scale quadrature with deep neural networks to compute numerical solutions while providing rigorous theoretical guarantees. The framework is applicable to continuous, discrete, and hybrid time scales within a unified computational setting.

The main features of this work include:

* Trapezoidal quadrature discretization on arbitrary time scales.
* Finite-dimensional fixed-point formulation.
* Neural residual learning based on the discrete fixed-point equation.
* Existence and uniqueness of the discrete solution.
* Consistency, stability, and convergence analysis.
* Residual-to-error estimate.
* Total error estimate separating discretization and learning errors.

---

## Repository Structure

```text
.
├── example1_volterra_neural.py
├── example2_volterra_neural.py
├── requirements.txt
├── README.md
└── figures/
```

---

## Requirements

* Python 3.8+
* PyTorch
* NumPy
* Matplotlib
* Pandas

Install the required packages using

```bash
pip install -r requirements.txt
```

or

```bash
pip install torch numpy matplotlib pandas
```

---

## Running the Examples

Example 1

```bash
python example1_volterra_neural.py
```

Example 2

```bash
python example2_volterra_neural.py
```

Each program

* computes the reference solution,
* trains the neural residual network,
* reports the residual loss, maximum error, and RMSE,
* generates numerical tables,
* saves convergence and solution figures.

---

## Mathematical Framework

The considered nonlinear Volterra dynamic integral equation is

[
x(t)=g(t)+\int_a^tK(t,s,x(s)),\Delta s,
\qquad
t\in[a,b]_{\mathbb T},
]

where (\mathbb T) is an arbitrary time scale.

The Volterra operator is discretized using a trapezoidal quadrature formula constructed from time-scale monomials, producing a finite-dimensional fixed-point problem

[
X=\Phi(X).
]

A deep neural network is trained by minimizing the residual of the discrete fixed-point equation

[
\mathcal L(\theta)
==================

\frac1N
\sum_{i=1}^{N}
\left|
X_\theta(t_i)-\Phi(X_\theta)(t_i)
\right|^2.
]

---

## Theoretical Results

The proposed framework establishes

* existence and uniqueness of the discrete solution,
* second-order consistency,
* stability,
* convergence,
* residual-to-error estimate,
* total error estimate

of the form

[
|X_\theta-x|_\infty
\le
C_1\Delta^2
+
C_2\sqrt{\mathcal L(\theta)}.
]

This estimate separates the deterministic discretization error from the neural approximation error.

---

## Numerical Results

The numerical experiments demonstrate

* rapid convergence of the residual loss,
* excellent agreement between numerical and exact solutions,
* approximation errors approaching machine precision,
* stable performance on continuous, discrete, and hybrid time scales.

---

## Citation

If you use this code in your research, please cite

```bibtex
@article{KhuddushGeorgiev2026,
  author  = {Mahammad Khuddush and Svetlin G. Georgiev},
  title   = {Deep Residual Learning for Volterra Dynamic Integral Equations on Arbitrary Time Scales},
  journal = {Under Review},
  year    = {2026}
}
```

---


---

## Contact

**Mahammad Khuddush**

Department of Mathematics

Vignan's Institute of Information Technology (Autonomous)

Visakhapatnam, Andhra Pradesh, India

E-mail: khuddush89@gmail.com

For questions, suggestions, or bug reports, please open a GitHub issue.
