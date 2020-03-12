import numpy as np
import typing as t


def apply_for_each_key(data: dict, func: t.Callable) -> dict:
    return {k:func(data[k]) for k in data}


def get_random_message(messages: t.List[str]) -> str:
    i = np.random.randint(low=0, high=len(messages))
    return messages[i]
