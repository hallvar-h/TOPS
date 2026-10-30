"""InterfacerDirect: a base class for what is called by the simulator after every step."""
from types import SimpleNamespace

import tops.dynamic as dps
from tops.simulator import InterfacerDirect, Simulator


class Counter(InterfacerDirect):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = 0

    def update(self, input):
        self.calls += 1


def test_an_interfacer_given_a_simulator_registers_itself_under_its_name():
    rts = SimpleNamespace(interface_functions={})
    interfacer = InterfacerDirect(rts)
    assert rts.interface_functions == {'InterfacerDirect': interfacer.interface_fun}


def test_the_name_can_be_chosen():
    rts = SimpleNamespace(interface_functions={})
    interfacer = InterfacerDirect(rts, name='controller')
    assert interfacer.interface_name == 'controller'
    assert rts.interface_functions == {'controller': interfacer.interface_fun}


def test_a_registered_interfacer_is_called_after_every_step_of_a_simulation():
    import tops.ps_models.k2a as model_data
    ps = dps.PowerSystemModel(model=model_data.load())
    ps.init_dyn_sim()
    sim = Simulator(ps, dt=5e-3, t_end=0.05)
    steps = []
    sim.interface_functions['steps'] = lambda sim: steps.append(sim.sol.t)
    interfacer = Counter(sim)
    sim.main_loop()
    assert interfacer.calls == len(steps) >= 10


def times(interfacer, instants):
    """The instants at which the interfacer updates, when the simulator is at each of `instants`."""
    rts = SimpleNamespace(sol=SimpleNamespace(t=0.0), interface_functions={})
    updated = []
    interfacer.update = lambda input: updated.append(rts.sol.t)
    for t in instants:
        rts.sol.t = t
        interfacer.interface_fun(rts)
    return updated


INSTANTS = [0, 0.03, 0.06, 0.1, 0.14, 0.2, 0.25]


def test_without_fs_the_interfacer_updates_at_every_step():
    assert times(InterfacerDirect(), INSTANTS) == INSTANTS


def test_fs_limits_the_updates_to_that_many_per_second_of_simulated_time():
    assert times(InterfacerDirect(fs=10), INSTANTS) == [0, 0.1, 0.2]


def test_a_subclass_may_set_fs_after_the_constructor():
    class Slow(InterfacerDirect):
        def __init__(self):
            super().__init__()
            self.fs = 10

    assert times(Slow(), INSTANTS) == [0, 0.1, 0.2]


def test_fs_goes_with_a_simulator_given_to_the_constructor():
    rts = SimpleNamespace(interface_functions={})
    interfacer = InterfacerDirect(rts, name='slow', fs=5)
    assert interfacer.fs == 5 and list(rts.interface_functions) == ['slow']
