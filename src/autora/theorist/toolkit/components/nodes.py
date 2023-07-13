from typing import List, Callable
from inspect import signature
from src.autora.theorist.toolkit.components.primitives import Arithmetic

###############
#       Objects
###############


###
# TreeNode
###
# attributes:
#   parent: Node object
#   offspring: list of Node objects
class TreeNode:

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

    # TODO: update __repr__ to collect node and all descendants
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


class Symbol(TreeNode):

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
