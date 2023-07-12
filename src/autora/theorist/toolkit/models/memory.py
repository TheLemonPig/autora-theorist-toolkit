from typing import List


class Stack:

    def __init__(self, stack: List = None, limit: int = 10):
        self._list = list() if stack is None else stack
        self._max = limit

    def pop(self):
        try:
            return self._list.pop()
        except IndexError:
            raise IndexError('No elements in stack to pop')

    def push(self, e):
        self._list.append(e)
        if self.size() > self._max:
            self._list = self._list[1:]

    def size(self):
        return len(self._list)

    def empty(self):
        return self.size() == 0

    def top(self):
        if self.empty():
            return None
        else:
            return self._list[-1]
