"""Residues for placing power system stabilisers (PSSs) in the two-area system, and the modes with and without them.

    uv run python examples/modal_analysis/pss_residues.py            # shows the plots
    uv run python examples/modal_analysis/pss_residues.py --save residues.png

A PSS measures the speed of a generator and adds a signal to the voltage reference of its AVR (the input v_pss). The
residue of a mode, from v_pss of an AVR to the speed of its generator, tells how much a PSS there can move the mode:
a small change of gain moves the eigenvalue by the residue times the PSS transfer function at the eigenvalue. The
magnitude shows where a PSS is most effective. The angle shows the phase the PSS must add for the eigenvalue to move
straight to the left: 180 degrees minus the angle of the residue.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

import tops.dynamic as dps
import tops.modal_analysis as dps_mdl
import tops.plotting as dps_plt


def swing_modes(lin, ps):
    """Electromechanical modes where generators swing against each other, sorted by frequency (inter-area first).

    Modes in the same frequency range where all generators move together (in phase) are left out. These are common
    modes of the system frequency, given by the governors, which a PSS is not meant to damp.
    """
    speed_idx = ps.gen.GEN.state_idx_global["speed"]
    idx = lin.get_mode_idx(["em", "non_conj"], damp_threshold=np.inf)
    shapes = lin.rev[speed_idx][:, idx]
    shapes = shapes / shapes[np.argmax(abs(shapes), axis=0), np.arange(len(idx))]  # Largest element real
    swing = np.max(abs(np.angle(shapes, deg=True)), axis=0) > 45
    idx = idx[swing]
    return idx[np.argsort(lin.freq[idx])]


def main(save=None):
    import tops.ps_models.k2a as model_data

    # Without PSSs
    model = model_data.load()
    model.pop("pss")
    ps = dps.PowerSystemModel(model=model)
    ps.init_dyn_sim()
    gen_names = ps.gen.GEN.par["name"]

    # From the PSS input of each AVR to the speed of each generator. The AVRs and generators are in the same order.
    lin = dps_mdl.PowerSystemModelLinearization(ps)
    lin.linearize(inputs=[ps.avr.SEXS.v_pss], outputs=[ps.gen.GEN.speed], get_eigs=True)
    print("Inputs: ", lin.input_desc)
    print("Outputs:", lin.output_desc)

    modes = swing_modes(lin, ps)
    print("\nElectromechanical (swing) modes without PSS:")
    for i in modes:
        print(f"  {lin.freq[i]:.3f} Hz, damping {100 * lin.damping[i]:5.1f} %")

    i_ia = modes[0]
    residues = np.diag(lin.residues(i_ia))  # From each AVR to its own generator
    print(
        f"\nResidues of the inter-area mode ({lin.freq[i_ia]:.3f} Hz), from v_pss of each AVR to its generator's speed:"
    )
    for name, r in zip(gen_names, residues):
        compensation = (180 - np.angle(r, deg=True) + 180) % 360 - 180
        print(
            f"  {name}: |R| = {abs(r):.2e}, angle {np.angle(r, deg=True):7.1f} deg, PSS phase needed {compensation:6.1f} deg"
        )

    # Check: close the loop through a small gain k (a PSS without phase compensation) at one generator at a time.
    # The eigenvalue moves by k times the residue, so with a pure gain, G3 and G4 would reduce the damping.
    k = 0.5
    print(f"\nShift of the inter-area eigenvalue with a gain of {k} from speed to v_pss (eigenvalues of A + B K C):")
    for j, name in enumerate(gen_names):
        gain = np.zeros((len(gen_names),) * 2)
        gain[j, j] = k
        eigs_cl = np.linalg.eigvals(lin.a + lin.b @ gain @ lin.c)
        shift = eigs_cl[np.argmin(abs(eigs_cl - lin.eigs[i_ia]))] - lin.eigs[i_ia]
        print(f"  {name}: actual {shift:.5f}, predicted {k * residues[j]:.5f}")

    # With PSSs, as in the model data
    ps_pss = dps.PowerSystemModel(model=model_data.load())
    ps_pss.init_dyn_sim()
    lin_pss = dps_mdl.PowerSystemModelLinearization(ps_pss)
    lin_pss.eigenvalue_decomposition()
    print("\nElectromechanical (swing) modes with PSS:")
    for i in swing_modes(lin_pss, ps_pss):
        print(f"  {lin_pss.freq[i]:.3f} Hz, damping {100 * lin_pss.damping[i]:5.1f} %")

    fig = plt.figure(figsize=(10, 4.5))
    ax_eigs = fig.add_subplot(1, 2, 1)
    ax_eigs.scatter(lin.eigs.real, lin.eigs.imag, marker="x", label="Without PSS")
    ax_eigs.scatter(
        lin_pss.eigs.real, lin_pss.eigs.imag, marker="o", facecolors="none", edgecolors="C1", label="With PSS"
    )
    ax_eigs.axvline(0, color="k", linewidth=0.5)
    ax_eigs.set_xlim(-2, 0.5)
    ax_eigs.set_ylim(0, 10)
    ax_eigs.set_xlabel("Real part [1/s]")
    ax_eigs.set_ylabel("Imaginary part [rad/s]")
    ax_eigs.set_title("Eigenvalues")
    ax_eigs.legend()

    ax_res = fig.add_subplot(1, 2, 2, projection="polar")
    # Scaled, but not rotated (as normalize=True would do), since the angles matter here
    dps_plt.plot_mode_shape(residues / abs(residues).max(), ax=ax_res)
    ax_res.set_rlim(0, 1.05)
    ax_res.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{np.rad2deg(x):.0f}°"))
    ax_res.set_title(f"Residues of the {lin.freq[i_ia]:.2f} Hz mode\n(G1 red, G2 blue, G3 green, G4 purple)")
    fig.tight_layout()

    if save:
        fig.savefig(save)
    else:
        plt.show()
    return lin, lin_pss, residues


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--save", help="write the figure to this file instead of showing it")
    main(save=parser.parse_args().save)
