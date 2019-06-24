def execute_callbacks(callbacks, arg_dict=None):
    if callbacks is None:
        return

    if arg_dict is None:
        for cb in callbacks:
            cb()
    else:
        for cb in callbacks:
            cb(arg_dict)
