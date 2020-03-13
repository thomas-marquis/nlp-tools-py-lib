import pytest

from nlp_tools import utils


def test_apply_for_each_key():
    input_dict = {
        'key1': 1,
        'key2': 2
    }

    def func(x):
        return x * 2

    expected = {
        'key1': 2,
        'key2': 4
    }
    res = utils.apply_for_each_key(input_dict, func)
    assert res == expected
