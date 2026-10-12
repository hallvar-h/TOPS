"""Loading and normalization of power system model data."""
import json
import os
import warnings

import tops.dyn_models as mdl_lib

PS_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ps_models')

# Alternative names for model categories
CATEGORY_ALIASES = {'transformers': 'trafos', 'generators': 'gen'}

# Model used for categories given as a plain table instead of {model_name: table}
DEFAULT_MODELS = {'lines': 'Line', 'trafos': 'Trafo', 'loads': 'Load', 'shunts': 'Shunt'}


def read_json(path):
    """Read model data from a JSON file, looking in the bundled ps_models folder if the path does not exist."""
    for candidate in [path, os.path.join(PS_MODELS_DIR, path)]:
        try:
            with open(candidate) as f:
                return json.load(f)
        except IOError:
            continue
    raise FileNotFoundError(f'Power system model data file not found: {path}')


def load_model_data(model):
    """Return a normalized (shallow) copy of model data.

    :param model: Dict with model data, or path to a .json file (which can also be the name of a bundled model,
        e.g. 'k2a.json').
    """
    if isinstance(model, str) and model.endswith('.json'):
        model_data = read_json(model)
    elif isinstance(model, dict):
        model_data = model
    else:
        raise TypeError(f'model must be a dict or a path to a .json file, not {type(model).__name__}')

    model_data = model_data.copy()

    for alias, key in CATEGORY_ALIASES.items():
        if alias in model_data:
            model_data[key] = model_data.pop(alias)

    for key, default_mdl in DEFAULT_MODELS.items():
        if key in model_data and isinstance(model_data[key], list) and len(model_data[key]) > 1:
            model_data[key] = {default_mdl: model_data[key]}

    return model_data


def find_model_class(category_key, mdl_key, user_mdl_lib=None):
    """Look up a model class, first in the user model library and then in the standard one.

    Returns None (with a warning) if the model is not found.
    """
    for lib in [user_mdl_lib, mdl_lib]:
        category = getattr(lib, category_key, None)
        if category is not None and hasattr(category, mdl_key):
            return getattr(category, mdl_key)

    warnings.warn(f'Model {category_key}:{mdl_key} not found in model library.')
    return None
