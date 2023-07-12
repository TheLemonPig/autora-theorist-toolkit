import numpy as np
from random import random


def less_than(a, b):
    return a < b


def remove_root(mdl_change):
    return np.log(random()) < -mdl_change


def add_root(mdl_change, num_rr):
    if num_rr == 0:
        return np.log(random()) < 0
    else:
        return np.log(random()) < (-mdl_change * num_rr)


def replace_node(mdl_change):
    return np.log(random()) < -mdl_change


# def replace_elementary_tree(mdl_change, omega_f, omega_i, nif, nfi, sf, si):
#     return np.log(random()) < (np.log(nif * omega_i * sf/(nfi * omega_f * si)) - mdl_change)

# Define a move
# Define its p(accept) rule


def bms_accept():
    ...


rulebook = dict({
    'None/Root': add_root,
    'Root/None': remove_root,
    'Node/Node': replace_node,
    'Node/Leaf': replace_node,
    'Leaf/Node': replace_node,
    # 'Node/Leaf': replace_elementary_tree,
    # 'Leaf/Node': replace_elementary_tree,
    'Leaf/Leaf': replace_node,
})

# if __name__ == '__main__':
#     print(less_than(3, 4))
#     print(less_than)
