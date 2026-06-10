# MAOOAM & qgs: Model Context

## 1. MAOOAM Overview

**MAOOAM** (Modular Arbitrary-Order Ocean-Atmosphere Model) is a low-order coupled ocean–atmosphere model formulated as a finite set of ODEs derived via spectral (Galerkin) truncation of the quasi-geostrophic equations on a beta-plane, with thermodynamic coupling between layers.

**Primary reference:** De Cruz, L., Demaeyer, J., & Vannitsem, S. (2016). The Modular Arbitrary-Order Ocean-Atmosphere Model: MAOOAM v1.0. *Geoscientific Model Development*, 9(8), 2793–2808. https://doi.org/10.5194/gmd-9-2793-2016

**Implementations:**
- Original Fortran/Lua: https://github.com/Climdyn/MAOOAM (also contains a legacy Python translation originally developed for use with DAPPER)
- Python (primary, actively maintained): qgs package — see §5

**Atmospheric foundations:** Charney & Straus (1980), Reinhold & Pierrehumbert (1982), Cehelsky & Tung (1987), all *J. Atmos. Sci.*
**Ocean foundation:** Pierini (2011/2012), Barsugli & Battisti (1998).

---

## 2. Equations of Motion

MAOOAM solves two coupled QG layers (atmosphere) over a shallow-water ocean layer. The atmosphere has a baroclinic structure (two pressure levels: 250 hPa, $\psi_a^1$; 750 hPa, $\psi_a^3$); the ocean is single-layer.

**Atmospheric vorticity (two levels, from De Cruz et al. 2016, Eqs. 1–2; also Tondeur et al. 2020, Eqs. 11–12):**

$$\frac{\partial}{\partial t}(\nabla^2 \psi_a^1) + J(\psi_a^1, \nabla^2 \psi_a^1) + \beta \frac{\partial \psi_a^1}{\partial x} = -k'_d \nabla^2(\psi_a^1 - \psi_a^3) + \frac{f_0}{\Delta p}\omega$$

$$\frac{\partial}{\partial t}(\nabla^2 \psi_a^3) + J(\psi_a^3, \nabla^2 \psi_a^3) + \beta \frac{\partial \psi_a^3}{\partial x} = +k'_d \nabla^2(\psi_a^1 - \psi_a^3) - \frac{f_0}{\Delta p}\omega - k_d\nabla^2(\psi_a^3 - \psi_o)$$

where $\omega$ is the vertical velocity at 500 hPa, $k_d$ is surface friction, $k'_d$ is interior (baroclinic) friction.

**Ocean vorticity:**
$$\frac{\partial}{\partial t}(\nabla^2 \psi_o - F_o \psi_o) + J(\psi_o, \nabla^2 \psi_o - F_o \psi_o) + \beta \frac{\partial \psi_o}{\partial x} = -r \nabla^2 \psi_o + \frac{C}{\delta} \nabla^2 (\psi_a^3 - \psi_o)$$

**Temperature equations** for atmosphere ($\theta_a$) and ocean ($T_o$) include radiative forcing, sensible heat flux (with exchange coefficient $\lambda$), and long-wave radiation balance. The radiative terms are linearized around a reference temperature — this is what makes the system at most **bilinear** (quadratic) in the state variables.

Full derivation: De Cruz et al. (2016), Secs. 2–3 and Appendix A.

---

## 3. Tensor Formulation of the RHS

This is a structurally important property for any differentiable reimplementation.

After Galerkin projection, the full system of $N = 2(n_a + n_o)$ ODEs takes the form (De Cruz et al. 2016, Eq. 17–18):

$$\frac{d\eta_i}{dt} = c_i + \sum_{j=1}^{N} m_{i,j}\,\eta_j + \sum_{j,k=1}^{N} t_{i,j,k}\,\eta_j\,\eta_k$$

Introducing a dummy variable $\eta_0 \equiv 1$, this collapses to a single tensor contraction:

$$\frac{d\eta_i}{dt} = \sum_{j=0}^{N}\sum_{k=0}^{N} \mathcal{T}_{i,j,k}\,\eta_j\,\eta_k \qquad (1 \le i \le N)$$

where $\mathcal{T}$ is a **sparse, upper-triangular (in last two indices) third-order tensor** of precomputed coefficients. The nonlinearity is **purely quadratic** — no transcendental functions appear in the tendency.

The tensor $\mathcal{T}$ is stored as a coordinate list of tuples $(i,j,k,\mathcal{T}_{i,j,k})$ and computed once at initialization from the spectral inner products of the basis functions (Appendix B of De Cruz et al. 2016).

**Implication for AD:** The RHS at runtime reduces to a sparse tensor-vector-vector contraction, which is straightforwardly expressible in JAX/PyTorch with `jnp.einsum` or equivalent; no symbolic manipulation is needed at differentiation time.

---

## 4. Spectral Basis & Truncation

Variables are expanded on Fourier basis functions over the rectangular domain $[0, 2\pi/n] \times [0, \pi]$:

- **Atmosphere:** $F_{Pi}(x,y)$ — products of sine and cosine functions satisfying zonally periodic / meridionally free-slip BCs. Mode types: A (purely meridional), K (zonal+meridional), L (alternate).
- **Ocean:** $\phi_{Hi}(x,y)$ — sine-based modes satisfying closed-basin (no-flux) BCs.

The **truncation** is specified by the maximum zonal/meridional wavenumbers for each component: atmosphere $(n_{a}^x, n_{a}^y)$ and ocean $(n_{o}^x, n_{o}^y)$, written compactly as a 4-number string. The state dimension is $N = 2n_a + 2n_o$, where $n_a$ = number of atmospheric spatial modes (used for **both** streamfunction $\psi_a$ and temperature $\theta_a$) and $n_o$ = number of oceanic spatial modes (used for both $\psi_o$ and $\delta T_o$).

| Config string | atm modes $n_a$ | ocean modes $n_o$ | $N$ (state dim) | $N+1$ (with $\eta_0$) |
|---------------|-----------------|-------------------|-----------------|------------------------|
| **2-2-2-4** | 10 | 8 | **36** | 37 |
| 2-2-2-4 (+atm) | 14 | 16 | 56 | 57 |
| (various) | — | — | 52 | 53 |

**The standard benchmark is the 36-dimensional model** (atmosphere truncated at wavenumber 2 in x and y → 10 modes; ocean at wavenumber 2 in x, 4 in y → 8 modes). This is the "DDV2016" reference configuration used in most published work (Vannitsem et al. 2015; Tondeur et al. 2020) and is what `qgs_maooam.py` produces by default. In qgs this is set with `set_atmospheric_channel_fourier_modes(2, 2)` and `set_oceanic_basin_fourier_modes(2, 4)` — see §6.3.

**Caution on config strings:** the 4-number notation is used inconsistently across the literature (the ordering of which number applies to x vs. y, and atmosphere vs. ocean, is not standardized). Several secondary sources transcribe it incorrectly. Always cross-check against the explicit mode count ($n_a$, $n_o$) and the resulting state dimension $N$ rather than relying on the string alone. The 52- and 56-dim configurations add ocean and/or atmosphere modes respectively (Tondeur et al. 2020 give exact breakdowns). Mode enumeration: De Cruz et al. (2016), Appendix B.

---

## 5. Key Physical Parameters

| Symbol | Description | Typical value |
|--------|-------------|---------------|
| $n$ | Domain aspect ratio ($2L_y/L_x$) | 1.5 |
| $f_0$ | Coriolis parameter (45°N) | $1.032 \times 10^{-4}$ s⁻¹ |
| $\beta$ | Meridional gradient of $f$ | $1.62 \times 10^{-11}$ m⁻¹ s⁻¹ |
| $C$ | Wind stress coupling coefficient | $O(0.01)$ kg m⁻² s⁻¹ |
| $\lambda$ (`hlambda`) | Ocean–atmosphere heat transfer coefficient | $\sim 10$–$20$ W m⁻² K⁻¹ |
| $k_d$ (`kd`) | Atmospheric surface friction | nondimensional, $O(10^{-2})$ |
| $k'_d$ (`kdp`) | Atmospheric interior friction | nondimensional, $O(10^{-2})$ |
| $r$ | Ocean friction | nondimensional, $O(10^{-7})$ |
| $d$ | Ocean–atmosphere mechanical coupling | nondimensional, $O(10^{-7})$ |

**Note:** many qgs parameters (`kd`, `kdp`, `r`, `d`) are **nondimensional** as stored, not in day⁻¹. The concrete parameter set from the standard `qgs_maooam.py` example (a strong-coupling / strong-LFV 36-dim configuration) is:

```python
model_parameters.set_params({'kd': 0.0290, 'kdp': 0.0290, 'n': 1.5,
                             'r': 1.e-7, 'h': 136.5, 'd': 1.1e-7})
model_parameters.atemperature_params.set_params({'eps': 0.7, 'T0': 289.3, 'hlambda': 15.06})
model_parameters.gotemperature_params.set_params({'gamma': 5.6e8, 'T0': 301.46})
model_parameters.atemperature_params.set_insolation(103.3333, 0)
model_parameters.gotemperature_params.set_insolation(310., 0)
```

**LFV parameter sensitivity:** the presence and amplitude of low-frequency variability depends critically on the wind stress coupling $C$. Vannitsem et al. (2015) and later work distinguish weak-LFV ($C \approx 0.01$) and strong-LFV ($C \approx 0.016$) regimes. The `hlambda` = 15.06 in the example above corresponds to a strong-coupling configuration.

**Model time unit (MTU):** time is nondimensionalized by $1/f_0$, so **1 MTU $= 1/f_0 \approx 9690$ s $\approx 0.112$ days $\approx 2.69$ hours**. (Equivalently, 10 MTU ≈ 1.1215 days.) Typical integration timesteps in the literature are 0.01–0.1 MTU (≈ 1.6–16 minutes).

---

## 6. qgs Python Package

**qgs** is the primary actively-maintained Python implementation of MAOOAM and related QG systems.

**Primary reference:** Demaeyer, J., De Cruz, L., & Vannitsem, S. (2020). qgs: A flexible Python framework of reduced-order multiscale climate models. *Journal of Open Source Software*, 5(56), 2597. https://doi.org/10.21105/joss.02597

**Repository:** https://github.com/Climdyn/qgs  
**Docs:** https://qgs.readthedocs.io  
**Install:** `pip install qgs`

### 6.1 Key Features vs. Fortran MAOOAM

- Pure Python (NumPy/SciPy); tensor coefficients computed symbolically via `sympy` then evaluated numerically
- Modular: atmosphere and ocean/land components assembled independently
- Additional configurations beyond original MAOOAM: land surface, channel geometry
- Integration: built-in Runge-Kutta integrator (primary); Julia's `DifferentialEquations.jl` via `diffeqpy` as optional alternative
- Multiprocessing support (required `if __name__ == '__main__':` guard on macOS/Windows)
- Benchmarked against Fortran and Lua MAOOAM implementations (Demaeyer et al. 2020)

### 6.2 Core Architecture

```
qgs/
├── params/          # QgParams: model configuration and physical parameters
├── tensors/         # Precomputation of T_{i,j,k} tensor from inner products
├── integrators/     # RungeKuttaIntegrator (primary); diffeqpy bridge
├── functions/       # RHS evaluation (tensor contraction)
└── basis/           # Spectral basis function construction
```

The two main entry-point scripts in the repo root are:
- `qgs_maooam.py` — ocean-atmosphere (MAOOAM) configuration
- `qgs_rp.py` — atmosphere-only (Reinhold-Pierrehumbert) configuration

### 6.3 Instantiation Pattern

```python
from qgs.params.params import QgParams
from qgs.functions.tendencies import create_tendencies
from qgs.integrators.integrator import RungeKuttaIntegrator

params = QgParams()
params.set_atmospheric_channel_fourier_modes(2, 2)   # -> 10 atmospheric modes
params.set_oceanic_basin_fourier_modes(2, 4)          # -> 8 oceanic modes  (=> 36-dim model)
params.set_params({'kd': 0.0290, 'kdp': 0.0290, 'n': 1.5, 'r': 1.e-7,
                   'h': 136.5, 'd': 1.1e-7})
# ... set temperature/insolation params (see §5) ...

# create_tendencies returns the RHS function f AND its Jacobian Df
f, Df = create_tendencies(params)

integrator = RungeKuttaIntegrator()
integrator.set_func(f)
integrator.integrate(t0, t, dt, ic=ic, write_steps=1)
time, traj = integrator.get_trajectories()
```

Key points:
- `create_tendencies(params)` returns both the tendency function `f` **and its analytic Jacobian** `Df` (relevant to §8).
- The tensor $\mathcal{T}$ is computed once inside `create_tendencies` (analytic or symbolic mode) and cached.
- Inner-product mode is `'analytic'` by default (fast, closed-form, MAOOAM basis only) or `'symbolic'` (slower, uses SymPy, supports custom bases).
- qgs also provides `RungeKuttaTglsIntegrator` — a **tangent-linear and adjoint integrator** built from `Df`. This is directly relevant if the goal is sensitivities/gradients (see §8.1).

**State vector ordering:** atmospheric barotropic streamfunction modes → atmospheric temperature modes → ocean streamfunction modes → ocean temperature modes (for 36-dim: indices roughly 0–9, 10–19, 20–27, 28–35). The $\eta_0 \equiv 1$ dummy variable is internal to the tensor and is **not** part of the user-facing state vector.

**Note:** The API has evolved since the 2020 JOSS paper. The GitHub repo and `readthedocs` documentation are the authoritative current reference.

---

## 7. Dynamical Properties

- **Chaotic** in standard configurations; positive Lyapunov exponents computed in De Cruz et al. (2018, *Nonlinear Processes in Geophysics*, 25, 387–412) and Vannitsem & Lucarini (2016, *J. Geophys. Res. Atmos.*, 121).
- **Lyapunov dimension** (Kaplan–Yorke): O(10) for the 36-dimensional configuration; grows with truncation.
- **Covariant Lyapunov vectors (CLVs)** computed and analyzed in Tondeur et al. (2020, *Chaos*) and De Cruz et al. (2018).
- **Timescale separation:** atmosphere fast (days–weeks), ocean slow (years–decades). The ocean acts as a slow manifold.
- **Low-frequency variability (LFV):** a long-periodic attracting orbit with period O(years–decades) appears for sufficiently strong wind stress coupling $C$. Studied in Vannitsem et al. (2015, *Physica D*, 309, 71–85). The LFV arises from ocean–atmosphere coupling, not from internal atmospheric dynamics alone.
- **Baroclinic instability:** solar forcing drives a meridional temperature gradient → vertical wind shear → baroclinic eddies. The ocean transports heat to counteract.
- **Multiple regimes:** the system behavior (LFV presence, predictability) is sensitive to truncation and parameter choices; configurations are not always directly comparable across papers.

---

## 8. Automatic Differentiation: Structural Considerations

This section is relevant to reimplementing MAOOAM as a differentiable model.

### 8.1 Why MAOOAM is AD-friendly

The RHS (§3) is a **sparse quadratic tensor contraction** with precomputed coefficients. At integration time it involves only:
- Dense or sparse matrix-vector products
- Element-wise multiplications
- Additions

No transcendental functions, no conditionals on state values, no implicit solvers in the RHS itself. This means the computational graph is simple and well-conditioned for both forward-mode and reverse-mode AD.

**Important — the Jacobian is already analytic.** Because the tendency is exactly quadratic,
$$\frac{\partial}{\partial \eta_l}\left(\frac{d\eta_i}{dt}\right) = \sum_{k=0}^{N}\left(\mathcal{T}_{i,l,k} + \mathcal{T}_{i,k,l}\right)\eta_k$$
is available in closed form. **qgs already exposes this**: `create_tendencies` returns `Df`, and `RungeKuttaTglsIntegrator` integrates the tangent-linear and adjoint models. So a differentiable reimplementation does **not** primarily exist to obtain the state Jacobian — that already exists and is exact.

The real value-add of a JAX/PyTorch reimplementation is therefore:
- **Composability** with neural networks / differentiable optimizers (e.g. end-to-end training of a hybrid model or a learned parameterization).
- **Parameter gradients** $\partial(\cdot)/\partial\theta$ for physical parameters $\theta$ (see §8.4) — qgs's TL model differentiates w.r.t. state, not parameters.
- **JIT compilation and GPU/TPU execution** for large ensembles or long differentiable rollouts.
- **Higher-order derivatives** and arbitrary differentiable loss functions over trajectories.

Worth stating explicitly in any writeup, to avoid framing AD as solving a problem (the Jacobian) that qgs has already solved analytically.

### 8.2 Implementation strategy (JAX)

The natural approach:

```python
import jax.numpy as jnp

# Precompute tensor T_ijk (sparse COO or dense) from qgs, convert to jnp arrays
# State vector eta: shape (N,), dtype float64

def rhs(eta, t, T_dense):
    # Augment with dummy variable eta_0 = 1
    eta_aug = jnp.concatenate([jnp.array([1.0]), eta])  # shape (N+1,)
    # Tendency: d eta_i / dt = sum_{j,k} T[i,j,k] * eta_aug[j] * eta_aug[k]
    return jnp.einsum('ijk,j,k->i', T_dense, eta_aug, eta_aug)[1:]  # drop dummy row

# Jacobian via jax.jacfwd or jax.jacrev
from jax import jacfwd
J = jacfwd(rhs, argnums=0)(eta0, 0.0, T_dense)
```

For the sparse tensor (COO format from qgs), use `jax.experimental.sparse` or manually index with `jnp.index_update` / scatter operations. The dense contraction is simpler but memory-intensive for large truncations.

**64-bit precision:** JAX defaults to 32-bit. Enable 64-bit with:
```python
from jax import config
config.update("jax_enable_x64", True)
```
This is important for climate model accuracy (recommended: always enable for MAOOAM).

### 8.3 Integration

For differentiating through the ODE trajectory (e.g., 4D-Var, sensitivity analysis), use a JAX-compatible ODE solver:
- `diffrax` (Kidger 2021) — the standard choice; supports adjoint sensitivity methods
- `jax.experimental.ode.odeint` — simpler but less control

Adjoint-based gradient computation (O(1) memory in trajectory length) is preferred over direct backpropagation through the integrator for long trajectories; see Kidger et al. (2021, *NeurIPS*) for the `diffrax` approach.

### 8.4 Parameter gradients

Physical parameters ($\lambda$, $k_d$, $C$, etc.) enter $\mathcal{T}$ linearly or at most bilinearly. For differentiating with respect to parameters, $\mathcal{T}$ must itself be expressed as a differentiable function of the parameters — which requires either re-deriving the inner product expressions analytically or wrapping the qgs tensor-computation pipeline in JAX. This is the main engineering challenge; the inner product computations in qgs currently use `sympy` + NumPy, not JAX primitives.

---

## 9. Key Literature

| Reference | Topic |
|-----------|-------|
| De Cruz et al. (2016), *GMD* | Original MAOOAM description; tensor formulation |
| Demaeyer et al. (2020), *JOSS* | qgs Python package |
| Vannitsem & De Cruz (2014), *GMD*, 7, 649–662 | Precursor 24-variable model (OA-QG-WS v2) |
| Vannitsem et al. (2015), *Physica D*, 309, 71–85 | LFV and ocean-atmosphere coupling |
| Demaeyer & Vannitsem (2018), *Chaos* | Stochastic parameterizations in MAOOAM |
| De Cruz et al. (2018), *Nonlinear Processes in Geophysics* | Lyapunov instability, high-dimensional configs |
| Vannitsem & Lucarini (2016), *JGR Atmos.*, 121 | Lyapunov exponents and CLVs |
| Tondeur et al. (2020), *Chaos* | CLVs in MAOOAM; DA implications |

---

## 10. Notes & Caveats

- **Time units:** 1 MTU $= 1/f_0 \approx 0.112$ days $\approx 2.69$ hours (so 10 MTU ≈ 1.1215 days). A common error is to treat 1 MTU as ≈1.12 days — that is off by a factor of 10. Confirm against published trajectories.
- **State vector indexing:** 0-indexed in Python; mode ordering must be verified against the params object for any given truncation. The $\eta_0 \equiv 1$ dummy variable is internal to the tensor formulation and is **not** element 0 of the user-facing state vector.
- **Configuration comparability:** results across papers are not directly comparable if truncation, $C$, or $\lambda$ differ. Always report the full configuration.
- **qgs API evolution:** the package has changed since the 2020 JOSS paper. The GitHub repo is the authoritative reference for current method names and calling conventions.
- **Vannitsem et al. (2015) journal:** *Physica D: Nonlinear Phenomena*, **not** *J. Atmos. Sci.* (a common miscitation).
- **Sparse tensor in AD:** the COO representation from qgs is not natively a JAX primitive; conversion and potential densification is needed. For small truncations (N=36), dense tensors (37×37×37 ≈ 50k elements) are tractable.
- **qgs already has TL/adjoint:** `create_tendencies` returns the analytic Jacobian and `RungeKuttaTglsIntegrator` integrates the tangent-linear/adjoint model. A differentiable reimplementation's novelty lies in parameter gradients, ML composability, and accelerator execution — not in obtaining the state Jacobian. Frame accordingly.
