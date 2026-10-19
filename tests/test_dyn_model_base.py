import types

import numpy as np
import pytest

import tops.dynamic as dps
from tops.dyn_models.avr import SEXS
from tops.dyn_models.blocks import Gain, Integrator, TimeConstant
from tops.dyn_models.utils import AlgebraicLoopError, DAEModel, get_submodules, output, store_outputs


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


class CountingGain(Gain):
    calls = 0

    @output
    def output(self, x, v):
        self.calls += 1
        return 2*super().output(x, v)  # Overriding output calling super().output is not a loop


def connect_in_loop(a, b):
    a.input = lambda x, v: b.output(x, v)
    b.input = lambda x, v: a.output(x, v)


@pytest.mark.parametrize('store', [True, False])
def test_algebraic_loop_raises(store):
    a, b = Gain(K=np.ones(1)), Gain(K=np.ones(1))
    connect_in_loop(a, b)
    with pytest.raises(AlgebraicLoopError, match='algebraic loop'):
        if store:
            with store_outputs([a, b]):
                a.output(None, None)
        else:
            a.output(None, None)
    assert not a._outputs_in_progress and not b._outputs_in_progress
    assert not a._store_output and not a._stored_outputs


def test_loop_through_state_is_evaluated():
    gain, lag = Gain(K=2*np.ones(1)), TimeConstant(T=np.ones(1))
    connect_in_loop(gain, lag)
    with store_outputs([gain, lag]):
        assert gain.output(np.array([3.0]), None) == 6.0


def test_outputs_are_computed_once_per_evaluation():
    mdl = CountingGain(K=np.ones(1))
    mdl.set_input('input', 1.0)
    with store_outputs([mdl]):
        assert mdl.output(None, None) == 2.0
        assert mdl.output(None, None) == 2.0
    assert mdl.calls == 1
    with store_outputs([mdl]):
        mdl.output(None, None)
    assert mdl.calls == 2


def test_stored_output_is_not_a_view_of_states():
    mdl = Integrator(n_units=2)
    x = np.array([1.0, 2.0])
    with store_outputs([mdl]):
        mdl.output(x, None)[:] = 0
    assert x.tolist() == [1.0, 2.0]


@pytest.mark.parametrize('kwargs, match', [
    ({}, 'exactly one of'),
    ({'par': np.zeros(2, dtype=[('name', 'U4')]), 'n_units': 2}, 'got par and n_units'),
    ({'n_units': 2, 'K': np.ones(2)}, 'got n_units and parameters'),
])
def test_parameters_must_be_given_one_way(kwargs, match):
    with pytest.raises(TypeError, match=match):
        Gain(**kwargs)


@pytest.mark.parametrize('kwargs', [{'K': np.ones(2), 'T': np.ones(3)}, {'K': 1.0}])
def test_parameters_must_have_one_value_per_unit(kwargs):
    with pytest.raises(ValueError, match='equal length'):
        Gain(**kwargs)


class TwoStates(DAEModel):
    def state_list(self):
        return ['a', 'b']


def test_state_idx_matches_state_desc():
    mdl = TwoStates(n_units=3, first_state_idx=10)
    for state in ['a', 'b']:
        assert mdl.state_idx[state].tolist() == np.where(mdl.state_desc[:, 1] == state)[0].tolist()
    assert mdl.state_idx_global['b'].tolist() == [11, 13, 15]


def test_models_have_no_inputs_or_outputs_by_default():
    mdl = TwoStates(n_units=1)
    assert mdl.input_list() == [] and mdl.output_list() == []
    assert mdl._input_values.dtype.names == ()
    assert not hasattr(mdl, 'input')
    with pytest.raises(NotImplementedError, match='TwoStates has no output'):
        mdl.output(None, None)


def test_blocks_have_one_input_and_output():
    mdl = Gain(K=np.ones(1))
    assert mdl.input_list() == ['input'] and mdl.output_list() == ['output']


class SEXSWithoutOutputList(SEXS):
    def output_list(self):
        return []  # As for a user model class SEXS_mod(DAEModel, AVR), where DAEModel's default wins


def test_connected_outputs_are_initialized_without_output_list():
    import tops.ps_models.k2a as model_data
    model = model_data.load()
    model['avr'] = {'SEXSWithoutOutputList': model['avr']['SEXS']}
    user_lib = types.SimpleNamespace(avr=types.SimpleNamespace(SEXSWithoutOutputList=SEXSWithoutOutputList))
    ps = dps.PowerSystemModel(model=model, user_mdl_lib=user_lib)
    ps.init_dyn_sim()
    assert max(abs(ps.ode_fun(0, ps.x_0))) < 1e-8
