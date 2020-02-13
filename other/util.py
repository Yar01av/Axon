from typing import Sequence
import copy

def execute_callbacks(callbacks, arg_dict=None):
    """
    Execute each callback in the list

    :param callbacks: the callbacks to execute
    :param arg_dict: the arguments to pass to each callback (same dict is passed to each of the callbacks)
    :return: None
    """

    if callbacks is None:
        return

    if arg_dict is None:
        for cb in callbacks:
            cb()
    else:
        for cb in callbacks:
            cb(arg_dict)


def get_seq_diff(seq: Sequence):
    """
    Returns the difference between the first and the last elements of the sequence without type conversions

    :param seq: the list to take the difference of.
    :return: None
    """

    if len(seq) == 0:
        raise AssertionError("List is too short")

    return seq[-1] - seq[0]


def get_timeseq_diff(timeseq):
    """
    :param timeseq: sequence of datetimes
    :return: The difference in seconds between the first and the last elements of the argument
    """

    return get_seq_diff(timeseq).total_seconds()


def convert_none_to_list(param):
    if param is None:
        return []
    else:
        return param


# TODO remove the wrapper
class Variable:
    """
    It is a wrapper for Python's variables that can be used cleanly in lambda expressions
    """

    def __init__(self, init_value):
        self.value = init_value

    def modify_value(self, function, *args_to_function):
        self.value = function(*args_to_function)

    def get_value(self):
        """

        :return: a shallow copy of the stored value
        """

        return copy.copy(self.value)
