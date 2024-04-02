from __future__ import annotations
from typing import Set, Iterable, List, Any
import numpy as np


class Node:
    """Node class represents a node in a graph."""

    def __init__(self, **kwargs) -> None:
        """
        Initialize a Node object.

        Keyword arguments:
        kwargs -- arbitrary keyword arguments representing attributes of the node
        """
        self.__dict__.update(kwargs)

    @property
    def name(self) -> str:
        """Return the class name as a string."""
        return self.__class__.__name__

    def __str__(self) -> str:
        """Return a string representation of the node."""
        return f"{type(self).__name__}: {self.__dict__}"

    def __repr__(self) -> str:
        """Return a string representation of the node for debugging."""
        return f"{type(self).__name__}({self.__dict__})"


class Edge:
    """Edge class represents a directed edge in a graph."""

    def __init__(self, tail: Node, head: Node, **kwargs) -> None:
        """
        Initialize an Edge object.

        Parameters:
        tail -- the tail node of the edge
        head -- the head node of the edge
        kwargs -- arbitrary keyword arguments representing attributes of the edge
        """
        self.tail = tail
        self.head = head
        self.__dict__.update(kwargs)

    @property
    def name(self) -> str:
        """Return the class name as a string."""
        return self.__class__.__name__

    def __len__(self) -> int:
        """Return the number of nodes in the edge (always 2 for directed edge)."""
        return 2

    def __iter__(self) -> Iterable[Node]:
        """Iterate over the nodes in the edge."""
        yield from [self.tail, self.head]

    def __str__(self) -> str:
        """Return a string representation of the edge."""
        return f"{type(self).__name__}: {self.tail} -> {self.head}, {self.__dict__}"

    def __repr__(self) -> str:
        """Return a string representation of the edge for debugging."""
        return f"{type(self).__name__}({self.tail}, {self.head}, {self.__dict__})"


class Nodes(Set[Node]):
    """Nodes class represents a set of nodes in a graph."""

    def __init__(self, *nodes: Node) -> None:
        """
        Initialize a Nodes object.

        Parameters:
        nodes -- nodes to be added to the set
        """
        super().__init__(nodes)

    def get(self, attr: str) -> np.ndarray:
        """
        Get values of the specified attribute for all nodes.

        Parameters:
        attr -- attribute name

        Returns:
        A list of values of the specified attribute for all nodes.
        """
        return np.array([getattr(node, attr) for node in self])

    def set(self, attr: str, vals: Iterable) -> None:
        """
        Set values of the specified attribute for all nodes.

        Parameters:
        attr -- attribute name
        vals -- iterable of values to be set for the attribute
        """
        for node, val in zip(self, vals):
            setattr(node, attr, val)

    def __str__(self) -> str:
        """Return a string representation of the set of nodes."""
        return f"{type(self).__name__}: {[str(node) for node in self]}"

    def __repr__(self) -> str:
        """Return a string representation of the set of nodes for debugging."""
        return f"{type(self).__name__}({[repr(node) for node in self]})"


class Edges(Set[Edge]):
    """Edges class represents a set of edges in a graph."""

    def __init__(self, *edges: Edge) -> None:
        """
        Initialize an Edges object.

        Parameters:
        edges -- edges to be added to the set
        """
        super().__init__(edges)

    def get(self, attr: str) -> np.ndarray:
        """
        Get values of the specified attribute for all edges.

        Parameters:
        attr -- attribute name

        Returns:
        A list of values of the specified attribute for all edges.
        """
        return np.array([getattr(edge, attr) for edge in self])

    def set(self, attr: str, vals: Iterable) -> None:
        """
        Set values of the specified attribute for all edges.

        Parameters:
        attr -- attribute name
        vals -- iterable of values to be set for the attribute
        """
        for edge, val in zip(self, vals):
            setattr(edge, attr, val)

    def __str__(self) -> str:
        """Return a string representation of the set of edges."""
        return f"{type(self).__name__}: {[str(edge) for edge in self]}"

    def __repr__(self) -> str:
        """Return a string representation of the set of edges for debugging."""
        return f"{type(self).__name__}({[repr(edge) for edge in self]})"


class Graph:
    """Graph class represents a graph."""

    def __init__(self, nodes: Nodes, edges: Edges) -> None:
        """
        Initialize a Graph object.

        Parameters:
        nodes -- a set of nodes in the graph
        edges -- a set of edges in the graph
        """
        self.nodes = nodes
        self.edges = edges

    def __str__(self) -> str:
        """Return a string representation of the graph."""
        return f"{type(self).__name__}: Nodes={self.nodes}, Edges={self.edges}"

    def __repr__(self) -> str:
        """Return a string representation of the graph for debugging."""
        return f"{type(self).__name__}({self.nodes}, {self.edges})"


if __name__ == '__main__':
    # Example usage
    n0 = Node(key="n0")
    n1 = Node(key="n1")
    n2 = Node(key="n2")

    e0 = Edge(n0, n1, weight=5)
    e1 = Edge(n1, n2, weight=3)
    e2 = Edge(n2, n0, weight=7)

    g = Graph(Nodes(n0, n1, n2), Edges(e0, e1, e2))

    print(n0)
    print(e0)
    print(g.nodes)
    print(g.edges)
    print(g)
