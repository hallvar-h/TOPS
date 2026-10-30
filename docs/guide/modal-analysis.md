# Modal analysis

Small-signal (modal) analysis describes how the system responds to small
disturbances around an operating point. TOPS linearises the model
numerically and computes the eigenvalues, which give the frequency and
damping of each oscillation mode, and the eigenvectors, which show which
generators take part.

## Linearise and compute eigenvalues

`PowerSystemModelLinearization` takes an initialised `PowerSystemModel`.
`linearize()` builds the state matrix **A** by numerical differentiation
around `ps.x0`, and `eigenvalue_decomposition()` computes the eigenvalues
and eigenvectors.

```python exec="on" source="above" session="modal" result="text"
import numpy as np
import tops.dynamic as dps
import tops.modal_analysis as dps_mdl
import tops.ps_models.k2a as model_data

ps = dps.PowerSystemModel(model=model_data.load())
ps.init_dyn_sim()

lin = dps_mdl.PowerSystemModelLinearization(ps)
lin.linearize()
lin.eigenvalue_decomposition()

print(f"A is {lin.a.shape[0]} x {lin.a.shape[1]}")
print(f"Largest real part of an eigenvalue: {lin.eigs.real.max():.2e}")
```

A positive real part means an unstable mode. The value here is zero to
numerical precision: the system has zero eigenvalues, because only the
differences between the rotor angles matter, not their absolute values.

The results are stored on the object:

| Attribute | Content |
|---|---|
| `lin.a` | State matrix |
| `lin.eigs` | Eigenvalues |
| `lin.freq` | Frequency of each mode [Hz] |
| `lin.damping` | Damping ratio of each mode (0 to 1) |
| `lin.rev`, `lin.lev` | Right and left eigenvectors (as columns and rows) |

## Find the electromechanical modes

`get_mode_idx` filters the modes. `"em"` keeps modes between 0.1 and 3 Hz,
the range of rotor oscillations. `"non_conj"` keeps one of each complex
pair, and `damp_threshold` drops well-damped modes. The result is sorted
from least to most damped.

```python exec="on" source="above" session="modal" result="text"
mode_idx = lin.get_mode_idx(["em", "non_conj"], damp_threshold=0.3)

for i in mode_idx:
    print(f"mode {i:2d}: {lin.freq[i]:.3f} Hz, damping {100 * lin.damping[i]:.1f} %")
```

## Plot the eigenvalues

```python exec="on" source="above" session="modal" html="1"
--8<-- "plot.py"
import matplotlib.pyplot as plt
import tops.plotting as dps_plt

dps_plt.plot_eigs(lin.eigs)
plt.xlim(-5, 1)
plt.xlabel("Real part [1/s]")
plt.ylabel("Imaginary part [rad/s]")
plt.show()
```

In an interactive window, `plot_eigs` shows the frequency and damping of
the eigenvalue under the mouse pointer.

## Mode shapes

The *mode shape* of a mode is its right eigenvector, restricted to the
generator speeds. It shows how much each generator takes part (length) and
in which phase (angle). Generators that point in opposite directions swing
against each other.

```python exec="on" source="above" session="modal" html="1"
--8<-- "plot.py"
speed_idx = ps.gen.GEN.state_idx_global["speed"]
mode_shape = lin.rev[np.ix_(speed_idx, mode_idx)]

fig, axes = plt.subplots(1, len(mode_idx), subplot_kw={"projection": "polar"}, figsize=(9, 3.2))
for ax, i, shape in zip(axes, mode_idx, mode_shape.T):
    dps_plt.plot_mode_shape(shape, ax=ax, normalize=True)
    ax.set_title(f"{lin.freq[i]:.2f} Hz")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
plt.show()
```

The colours follow the generator order: `G1` red, `G2` blue, `G3` green,
`G4` purple. The two modes around 1.1 Hz are *local* modes: in each, the two
generators of one area swing against each other. In the slower *inter-area*
mode, G1 and G2 move together, as do G3 and G4, and the two pairs oscillate
relative to each other.

## Inputs and outputs

To get a full state-space model **(A, B, C)**, for example for controller
design, `linearize()` also accepts `input_description` and
`output_description`. Together with `residues(mode_idx)`, these give the
residues of a transfer function. See `tops/modal_analysis.py` and the
`examples/notebooks/modal_analysis.ipynb` notebook.
