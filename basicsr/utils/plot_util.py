import re


def read_data_from_txt_2v(path, pattern, step_one=False):
    """Read data from txt with 2 returned values (usually [step, value]).

    Args:
        path (str): path to the txt file.
        pattern (str): re (regular expression) pattern.
        step_one (bool): add 1 to steps. Default: False.
    """
    with open(path) as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    steps = []
    values = []

    pattern = re.compile(pattern)
    for line in lines:
        match = pattern.match(line)
        if match:
            steps.append(int(match.group(1)))
            values.append(float(match.group(2)))
    if step_one:
        steps = [v + 1 for v in steps]
    return steps, values


def read_data_from_txt_1v(path, pattern):
    """Read data from txt with 1 returned values.

    Args:
        path (str): path to the txt file.
        pattern (str): re (regular expression) pattern.
    """
    with open(path) as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    data = []

    pattern = re.compile(pattern)
    for line in lines:
        match = pattern.match(line)
        if match:
            data.append(float(match.group(1)))
    return data


def smooth_data(values, smooth_weight):
    """Smooth data using a first-order IIR low-pass filter.

    Args:
        values (list): A list of values to be smoothed.
        smooth_weight (float): Smooth weight.
    """
    values_sm = []
    last_sm_value = values[0]
    for value in values:
        value_sm = last_sm_value * smooth_weight + (1 - smooth_weight) * value
        values_sm.append(value_sm)
        last_sm_value = value_sm
    return values_sm
