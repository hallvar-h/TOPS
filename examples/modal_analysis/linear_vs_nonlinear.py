"""Step response of the linearised model, from a governor input to generator outputs, against the nonlinear simulation.

    uv run python examples/modal_analysis/linear_vs_nonlinear.py            # shows the plots
    uv run python examples/modal_analysis/linear_vs_nonlinear.py --save step.png
    uv run python examples/modal_analysis/linear_vs_nonlinear.py --step 0.01   # a larger step

The two agree for small steps. For larger steps they drift apart, mainly because the output of the PSSs hits its
limits in the nonlinear simulation.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

import tops.dynamic as dps
import tops.modal_analysis as dps_mdl
from tops.simulator import ResultKeeper, Simulator


def main(step=0.001, t_step=1, t_end=20, save=None):
    import tops.ps_models.k2a as model_data

    ps = dps.PowerSystemModel(model=model_data.load())
    ps.init_dyn_sim()
    gen, gov = ps.gen.GEN, ps.gov.TGOV1

    # Linearise, from the bias of governor GOV1 to the speed of all generators and the electrical power of G1
    lin = dps_mdl.PowerSystemModelLinearization(ps)
    a, b, c, d = lin.linearize(
        inputs=[(gov.bias, "GOV1")],
        outputs=[gen.speed, (gen.p_e, "G1")],
    )
    print("Inputs: ", lin.input_desc)
    print("Outputs:", lin.output_desc)

    # Linear step response. The outputs are deviations from the operating point, so their initial values are added.
    t = np.arange(0, t_end, 5e-3)
    u = step * (t >= t_step)
    _, dy, _ = signal.lsim(signal.StateSpace(a, b, c, d), u, t)
    y_lin = dy + np.concatenate([gen.speed(lin.x0, lin.v0), gen.p_e(lin.x0, lin.v0)[:1]])

    # Nonlinear simulation with the same step, applied with set_input
    sim = Simulator(ps, dt=5e-3, t_end=t_end)
    res = ResultKeeper(sim, speed=gen.speed, p_e=gen.p_e)
    sim.interface_functions["ResultKeeper"] = res.update
    bias_0 = gov._input_values["bias"][0]
    while sim.sol.t < t_end:
        if sim.sol.t >= t_step:
            gov.set_input("bias", bias_0 + step, 0)
        sim.make_simulation_step()
    speed = np.array(res.store["speed"])
    p_e = np.array(res.store["p_e"])[:, 0]

    fig, axes = plt.subplots(2, sharex=True, figsize=(8, 6))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, name in enumerate(gen.par["name"]):
        axes[0].plot(res.t, speed[:, i], color=colors[i], label=name)
        axes[0].plot(t, y_lin[:, i], color=colors[i], linestyle="--")
    axes[0].set_ylabel("Speed [p.u.]")
    axes[0].legend(title="Solid: nonlinear\nDashed: linear", ncol=2)
    axes[1].plot(res.t, p_e, color=colors[0], label="Nonlinear")
    axes[1].plot(t, y_lin[:, 4], color=colors[0], linestyle="--", label="Linear")
    axes[1].set_ylabel("Electrical power of G1\n[p.u., generator base]")
    axes[1].set_xlabel("Time [s]")
    axes[1].legend()
    fig.suptitle(f"Step of {step} in the bias of GOV1 at t = {t_step} s")
    fig.tight_layout()

    if save:
        fig.savefig(save)
    else:
        plt.show()
    return lin, (t, y_lin), (np.array(res.t), speed, p_e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--step", type=float, default=0.001, help="size of the step in the bias of GOV1")
    parser.add_argument("--save", help="write the figure to this file instead of showing it")
    args = parser.parse_args()
    main(step=args.step, save=args.save)
