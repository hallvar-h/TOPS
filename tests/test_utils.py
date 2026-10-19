import numpy as np
import pandas as pd
import tops.utility_functions as utils


def test_structured_array_from_dataframe():
    dataframe = pd.DataFrame.from_dict({"X": [1, 2, 3], "Y": ["A", "B", "C"]})
    struct_array = utils.structured_array_from_dataframe(dataframe)
    assert all(struct_array["X"] == [1, 2, 3])
    assert all(struct_array["Y"] == ["A", "B", "C"])


def test_structured_array_from_dataframe_matches_list():
    names, entries = ["name", "V_n"], [["A", 20.0], ["B", 300.0]]
    from_df = utils.structured_array_from_dataframe(pd.DataFrame(entries, columns=names))
    from_list = utils.structured_array_from_list(names, entries)
    assert from_df.dtype == from_list.dtype
    assert all(from_df == from_list)
