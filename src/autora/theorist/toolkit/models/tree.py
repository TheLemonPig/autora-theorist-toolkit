from src.autora.theorist.toolkit.components.nodes import Symbol, Parameter
from typing import List


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
        self._root: Symbol = Symbol()
        self._nodes: List[Symbol] = []
        self._leaves: List[Symbol] = []
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
    def replace(self, old_node: Symbol, new_node: Symbol, **kwargs):
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
        return len(set([self._root]+self._nodes+self._leaves))

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
