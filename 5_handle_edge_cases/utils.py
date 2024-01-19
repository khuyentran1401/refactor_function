from typing import Union, Any
from collections.abc import Iterable
import pandas as pd


def check_variables_is_list(variables: Union[Any, Iterable[Any]]) -> Iterable[Any]:
    if isinstance(variables, list):
        return variables
    return [variables]
