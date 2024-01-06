from typing import Union, TypeVar, Any
from collections.abc import Iterable
import pandas as pd

IntOrStr = TypeVar("IntOrStr", int, str)


def is_non_string_iterable(variable):
    return isinstance(variable, Iterable) and not isinstance(variable, str)


def check_variables_is_list(variables: Union[Any, Iterable[Any]]) -> Iterable[Any]:
    if isinstance(variables, list):
        return variables
    return [variables]


def check_variables_in_df(df: pd.DataFrame, variables: Union[Any, Iterable[Any]]):
    if isinstance(variables, (str, int)):
        if variables not in df.columns.to_list():
            raise KeyError(f"The feature {variables} is not in the dataframe.")
    else:
        missing_vars = set(variables) - set(df.columns)

        if missing_vars:
            missing_vars_str = ", ".join(str(var) for var in missing_vars)
            raise KeyError(
                f"The following features are not in the dataframe: {missing_vars_str}"
            )

    return variables
