"""The modal analysis examples with inputs and outputs must keep running."""

import importlib.util
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")


def load_example(name):
    path = Path(__file__).parent.parent / "examples" / "modal_analysis" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_linear_step_response_follows_the_nonlinear_simulation(tmp_path):
    lin, (t, y_lin), (t_nl, speed, p_e) = load_example("linear_vs_nonlinear").main(
        step=1e-4, t_end=5, save=tmp_path / "step.png"
    )
    y_lin = np.column_stack([np.interp(t_nl, t, y_lin[:, i]) for i in range(y_lin.shape[1])])
    assert abs(y_lin[:, :4] - speed).max() < 0.02 * abs(speed).max()
    assert abs(y_lin[:, 4] - p_e).max() < 0.02 * np.ptp(p_e)
    assert (tmp_path / "step.png").stat().st_size > 0


def test_residues_predict_the_shift_of_the_inter_area_mode(tmp_path):
    lin, lin_pss, residues = load_example("pss_residues").main(save=tmp_path / "residues.png")
    assert residues.shape == (4,)
    # The PSSs at G3 and G4 have the largest effect on the inter-area mode
    assert set(np.argsort(abs(residues))[-2:]) == {2, 3}
    assert (tmp_path / "residues.png").stat().st_size > 0
