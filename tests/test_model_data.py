from types import SimpleNamespace

import pytest

import tops.dyn_models as mdl_lib
from tops.model_data import load_model_data, find_model_class


def test_aliases_and_default_models_are_normalized():
    lines = [['name', 'from_bus', 'to_bus'], ['L1', 'B1', 'B2']]
    model = {'generators': {'GEN': []}, 'transformers': [['name'], ['T1']], 'lines': lines}
    data = load_model_data(model)

    assert 'generators' not in data and data['gen'] == {'GEN': []}
    assert 'transformers' not in data and data['trafos'] == {'Trafo': [['name'], ['T1']]}
    assert data['lines'] == {'Line': lines}


def test_input_dict_is_not_modified():
    model = {'generators': {'GEN': []}, 'lines': [['name'], ['L1']]}
    load_model_data(model)
    assert model == {'generators': {'GEN': []}, 'lines': [['name'], ['L1']]}


def test_bundled_json_by_name():
    assert 'buses' in load_model_data('k2a.json')


def test_find_standard_model():
    assert find_model_class('gen', 'GEN') is mdl_lib.gen.GEN


def test_user_model_takes_precedence():
    class GEN:
        pass
    user_lib = SimpleNamespace(gen=SimpleNamespace(GEN=GEN))
    assert find_model_class('gen', 'GEN', user_lib) is GEN
    # Models not in the user library are still found in the standard library
    assert find_model_class('avr', 'SEXS', user_lib) is mdl_lib.avr.SEXS


def test_missing_model_warns_and_returns_none():
    with pytest.warns(UserWarning, match='gen:NoSuchModel'):
        assert find_model_class('gen', 'NoSuchModel') is None
