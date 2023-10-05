# Nathaniel Linden 2023
# Utilities to help with the project
import jax.numpy as jnp

def construct_full_param_list(param_list, param_idxs, nominal_param_array):
    """ Constructs a full parameter list from a list of parameter indices and a dictionary of all nominal parameters in the model.
    
    We do this using jax.numpy to enable jit.

    All inputs must by arrays or jax arrays.
    """

    return nominal_param_array.at[param_idxs].set(param_list)


def parse_identifiability_results(path_to_ID_txt):

    file = open(path_to_ID_txt, 'r')
    lines = file.readlines()
    file.close()

    # parse the lines
    identifiabile = lines[1].strip(', \n')

    if len(lines) > 3:
        non_identifiable = lines[3].strip(', \n')
    else:
        non_identifiable = ''

    # split the strings
    identifiabile = identifiabile.split(', ')
    non_identifiable = non_identifiable.split(', ')

    # remove states from the lists
    identifiabile = [item for item in identifiabile if 'x' not in item]
    non_identifiable = [item for item in non_identifiable if 'x' not in item]

    return identifiabile, non_identifiable
    