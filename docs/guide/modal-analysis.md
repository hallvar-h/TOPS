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
| `lin.b`, `lin.c`, `lin.d` | Input, output and feedthrough matrices (see [Inputs and outputs](#inputs-and-outputs)) |
| `lin.input_desc`, `lin.output_desc` | Labels of the inputs and outputs |

The derivatives are central differences with relative step `eps` (default
`1e-6`), which can be set with `PowerSystemModelLinearization(ps, eps=...)`.

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

## Participation factors

The participation factor of a state in a mode measures how much the state
takes part in it. Each column of `participation_factors()` is one mode and
sums to one.

```python exec="on" source="above" session="modal" result="text"
p = abs(lin.participation_factors())
for i in mode_idx:
    top = np.argsort(p[:, i])[::-1][:4]
    print(f"{lin.freq[i]:.3f} Hz:", ", ".join(f"{u} {s}" for u, s in ps.state_desc[top]))
```

## Inputs and outputs

For a full state-space model **(A, B, C, D)**, for example for controller
design, pass inputs and outputs to `linearize()`. Each one is a function,
either alone (all units) or as `(function, units)`, with units selected by
name or index, one or a list:

- An input is an input of a model (from its `input_list()`), such as
  `ps.avr.SEXS.v_setp` or `ps.gov.TGOV1.bias`. Inputs of blocks work too,
  e.g. `ps.avr.SEXS.tg_red.input`. Take inputs after `ps.init_dyn_sim()`,
  which replaces the connected ones.
- An output is any function `f(x, v)`, such as `ps.gen.GEN.speed` or
  `lambda x, v: abs(v[3])`, as for `ResultKeeper.calculate`. Units can be
  selected by name only for functions of a model. For other functions, give
  the indices of the elements they return.

Each selected unit adds one input or one output. An input is perturbed by
adding a small signal to its value. This also works for connected inputs,
such as `P_m` of a generator with a governor, or `v_setp` of an AVR.

```python exec="on" source="above" session="modal" result="text"
a, b, c, d = lin.linearize(
    inputs=[(ps.avr.SEXS.v_setp, "AVR1")],
    outputs=[(ps.gen.GEN.speed, ["G1", "G3"])],
    get_eigs=True,
)
print("B:", b.shape, " C:", c.shape, " D:", d.shape)
print(lin.input_desc, lin.output_desc)
```

`residues(mode_idx)` gives the residues of the transfer functions from the
inputs to the outputs for one mode, with one row per output and one column
per input. A large residue means that the mode is easy to control from that
input and easy to observe in that output.

```python exec="on" source="above" session="modal" result="text"
i = lin.get_dominant_mode()
print(f"Dominant mode: {lin.freq[i]:.3f} Hz, damping {100 * lin.damping[i]:.1f} %")
print(abs(lin.residues(i)))
```

Outputs may be complex, such as the current `i` of a generator. In that case,
**C** and **D** are complex.
