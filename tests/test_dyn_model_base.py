import numpy as np
import pytest

import tops.dynamic as dps
from tops.dyn_models.blocks import Gain, TimeConstant
from tops.dyn_models.utils import DAEModel, get_submodules


class Inner(DAEModel):
    def add_blocks(self):
        self.lag = TimeConstant(T=np.ones(self.n_units))
        self.gain = Gain(K=np.ones(self.n_units))


class Outer(DAEModel):
    def add_blocks(self):
        self.second = Inner(n_units=self.n_units)
        self.first = Gain(K=np.ones(self.n_units))


def test_blocks_are_registered_in_creation_order():
    mdl = Outer(name=['A', 'B'])
    assert list(mdl.blocks) == ['second', 'first']
    assert get_submodules(mdl) == [mdl, mdl.second, mdl.second.lag, mdl.second.gain, mdl.first]


def test_blocks_are_named_after_attributes():
    mdl = Outer(name=['A', 'B'])
    assert list(mdl.first.par['name']) == ['A/first', 'B/first']
    assert list(mdl.second.lag.par['name']) == ['A/second/lag', 'B/second/lag']
    assert mdl.second.lag.state_desc.tolist() == [['A/second/lag', 'x'], ['B/second/lag', 'x']]
    assert mdl.second.lag.par['T'].tolist() == [1, 1]  # Other parameters are kept


def test_block_registration_follows_reassignment():
    mdl = Outer(n_units=1)
    mdl.first = None
    assert list(mdl.blocks) == ['second']
    with pytest.raises(ValueError, match='already assigned'):
        mdl.alias = mdl.second


def test_inputs_are_attached_per_instance():
    a, b = Gain(K=np.ones(2)), Gain(K=np.ones(2))
    b.set_input('input', 2.0)

    a.input = lambda x, v: np.array([5.0, 5.0])
    assert np.all(b.input(None, None) == 2.0)
    assert 'input' not in vars(Gain)


def test_disconnect_input_removes_connection():
    mdl = Gain(K=np.ones(2))
    mdl.set_input('input', 1.0)
    mdl.input = lambda x, v: np.array([5.0, 5.0])

    mdl.disconnect_input('input')
    assert np.all(mdl.input(None, None) == 1.0)


def test_set_input_affects_unconnected_units_of_connected_input():
    import tops.ps_models.k2a as model_data
    model = model_data.load()
    model['pss']['STAB1'] = model['pss']['STAB1'][:-1]  # No PSS on G4, so v_pss of G4 is unconnected
    ps = dps.PowerSystemModel(model=model)
    ps.init_dyn_sim()
    gen = ps.gen['GEN']
    pss = ps.pss['STAB1']
    x = ps.x_0.copy()
    x[pss.washout.state_idx_global['x']] = 0.01  # Non-zero PSS output

    gen.set_input('v_pss', 0.1)
    v_pss = gen.v_pss(x, ps.v_0)
    assert v_pss[3] == 0.1
    assert np.allclose(v_pss[:3], pss.output(x, ps.v_0))
    assert gen.v_pss(x, ps.v_0) is not v_pss  # A new array is returned on each call
