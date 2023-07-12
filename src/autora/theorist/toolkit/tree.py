from typing import List, Callable
from inspect import signature

from toolbox.theorist.primitives import Arithmetic


###############
#       Objects
###############


###
# TreeNode
###
# attributes:
#   parent: Node object
#   offspring: list of Node objects
class _TreeNode:

    def __init__(self, parent=None, children=None):
        self._parent: __class__.__name__ = parent
        self._children: List[__class__.__name__] = [] if children is None else children

    def __call__(self, x):
        raise TypeError('Uninitialized Node cannot be evaluated')

    def __getitem__(self, item):
        if isinstance(item, list):
            if len(item) > 0:
                return self.__getitem__(item[0]).__getitem__(item[1:])
            else:
                raise KeyError('Specified tree location does not exist')
        else:
            if item == -1:
                return self._parent
            else:
                return self._children[item]

    def __repr__(self):
        return '...'

    def get_parent(self):
        return self._parent

    def get_children(self):
        return self._children

    def get_family(self):
        if self.has_children():
            descendants = list(child.get_family() for child in self.get_children())
            family = [self]
            for subfamily in descendants:
                family += subfamily
            return family
        else:
            return [self]

    def has_children(self):
        return bool(self._children)

    def has_parent(self):
        return bool(self._parent)

    def is_filled(self):
        return False

    def is_initialized(self):
        return False

    def replace_child(self, old, new):
        self._children[self._children.index(old)] = new

    def replace_parent(self, parent):
        self._parent = parent


class Symbol(_TreeNode):

    def __init__(self, parent=None, children=None):
        super().__init__(parent, children)
        self._value = None
        self._label = 'R'

    def __repr__(self):
        return self._label

    def is_initialized(self):
        return self._value is not None

    def is_filled(self):
        return self.is_initialized() and all(children.is_filled() for children in self.get_children())

    def get_value(self):
        return self._value

    def args(self):
        return []


class Variable(Symbol):

    def __init__(self, value, parent=None):
        super().__init__(parent=parent)
        self._label = value
        self._value: int = int(value[2:-1])

    def __call__(self, x):
        try:
            return x[self._label]
        except KeyError:
            try:
                return x[self._value]
            except KeyError:
                raise KeyError('Data does not contain label for variable: ' +
                               self._label+', nor is large enough to imply')


class Parameter(Symbol):

    def __init__(self, value=0.0, parent=None):
        super().__init__(parent=parent, children=None)
        self._value: float = value

    def __call__(self, x, *parameters):
        return self.get_value()

    def label(self, label):
        self._label = label

    def set_value(self, num):
        self._value = num


class Operator(Symbol):

    def __init__(self, value, parent=None):
        super().__init__(parent=parent,
                         children=[Symbol(parent=self) for _ in signature(value).parameters])
        self._value: Callable = value
        self._label: str = str(value)

    def __repr__(self):
        if isinstance(self._value, Arithmetic):
            return '('+self._children[0].__repr__() + str(self._value) + self._children[1].__repr__()+')'
        else:
            return str(self._value) + '(' + ','.join(child.__repr__() for child in self.get_children()) + ')'

    def args(self):
        return signature(self._value).parameters


###
# Tree(Model)
###
# attributes:
#   root: Node object
#   value: function object with arguments for x and parameters
#   coefficients: List of floats
#
# methods:
# fit: calls the object-free fit function above and then updates its parameters
# predict: sklearn format for predict
# __call__: calls the object-oriented predict method above
#
class Tree:

    def __init__(self):
        super().__init__()
        self._root: _TreeNode = Symbol()
        self._nodes: List[_TreeNode] = []
        self._leaves: List[_TreeNode] = []
        self._parameters: List[Parameter] = []
        self._parameter_dict: dict[str, float] = {}
        self.catalog()

    def __repr__(self):
        return self._root.__repr__()

    def __getitem__(self, item):
        return self.get_all_nodes()[item]

    def __len__(self):
        return len(self.get_all_nodes())

    def set_parameters(self, parameter_values):
        for i, parameter in enumerate(self._parameters):
            parameter.set_value(parameter_values[i])
        self.catalog()

    def is_filled(self):
        return len(self.get_uninitialized()) == 0

    def get_uninitialized(self):
        uninitialized_nodes = []
        for node in self.get_all_nodes():
            if not node.is_initialized():
                uninitialized_nodes.append(node)
        return uninitialized_nodes

    def get_all_nodes(self):
        return [self.get_root()] + self.get_nodes() + self.get_leaves()

    def get_root(self):
        return self._root

    def get_nodes(self):
        return self._nodes

    def get_leaves(self):
        return self._leaves

    def get_parameters(self):
        return self._parameters

    def get_parameter_dict(self):
        return self._parameter_dict

    # depth-first search for uninitialized node
    def fill(self, old_node: Symbol, new_node: Symbol):
        if old_node.is_initialized():
            for node in old_node.get_children():
                self.fill(node, new_node)
        else:
            self.replace(old_node, new_node)

    # replace old node with new node
    def replace(self, old_node: _TreeNode, new_node: _TreeNode, **kwargs):
        if 'NR' in kwargs:
            assert len(old_node.get_children()) == len(new_node.get_children())
            for i, child in enumerate(old_node.get_children()):
                child.replace_parent(new_node)
                new_node.replace_child(old=new_node[i], new=child)
        else:
            if old_node.has_parent():
                old_node.get_parent().replace_child(old_node, new_node)
                new_node.replace_parent(old_node.get_parent())
            else:
                self._root = new_node
                new_node.replace_parent(None)
        self.catalog()

    # catalog all nodes into either root, leaves, or nodes (for inter-nodes)
    def catalog(self):
        all_nodes = self._root.get_family()
        self._clear()
        for node in all_nodes:
            if node.has_parent():
                if node.has_children():
                    self._nodes.append(node)
                else:
                    self._leaves.append(node)
            else:
                self._root = node
            if isinstance(node, Parameter):
                self._parameters.append(node)
                self._label_parameter(node)

    def size(self):
        return len(set(self._root+self._nodes+self._leaves))

    # reset node lists
    def _clear(self):
        self._root = None
        self._nodes = []
        self._leaves = []
        self._parameters = []
        self._parameter_dict = {}

    def _label_parameter(self, node: Parameter):
        digit_label = 0
        while '__a'+str(digit_label)+'__' in self._parameter_dict.keys():
            digit_label += 1
        node.label('__a'+str(digit_label)+'__')
        self._parameter_dict.update({str(node): node.get_value()})
