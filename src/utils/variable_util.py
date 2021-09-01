import re


def variables_to_names(variables):
    """
    Convert a list or tuple of Variable objects to list of their names.
    :param variables: a list or tuple of Variable objects.
    :return: list of `str`
    """
    names = []
    for var in variables:
        names.append(get_variable_name(var))
    return names


def get_variable_name(variable):
    """
    Convert Variable object to it's name.
    :param variable: Variable object.
    :return: `str` name of Variable object.
    """
    name = variable.name
    match = re.match("^(.*):\\d+$", name)
    if match is not None:
        name = match.group(1)
    return name
