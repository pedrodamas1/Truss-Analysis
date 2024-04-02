from typing import List, Tuple, Iterable, Dict
from scipy.sparse import csr_matrix
import networkx as nx
from graph import Node, Edge, Nodes, Edges, Graph
import matplotlib.pyplot as plt
import numpy as np


def adj_mat(nodes: List[Node], edges: List[Tuple[Node, Node]]) -> csr_matrix:
    """
    Returns the adjacency matrix (node-node) with dimension: n x n.

    Parameters:
    - nodes: List of nodes in the graph.
    - edges: List of edges in the graph represented as tuples (tail, head).

    Returns:
    - csr_matrix: Sparse adjacency matrix.
    """
    
    # Dimension of the adjacency matrix
    dim = len(nodes), len(nodes)
    
    # Dictionary to map nodes to their indices
    node_dict = {node: i for i, node in enumerate(nodes)}
    
    # Lists to store row, column, and data of the matrix
    row, col, data = [], [], []
    
    # Iterate over edges to populate the matrix
    for tail, head in edges:
        # Index of the tail node
        i = node_dict[tail]
        # Index of the head node
        j = node_dict[head]
        # Add entries for both directions (tail to head and head to tail)
        row.extend([i, j])
        col.extend([j, i])
        data.extend([1, 1])
        
    # Create a sparse matrix using the collected data
    return csr_matrix((data, (row, col)), shape=dim)


def inc_mat(nodes: List[Node], edges: List[Tuple[Node, Node]]) -> csr_matrix:
    """
    Returns the incidence matrix (node-edge) with dimension: n x e.

    Parameters:
    - nodes: List of nodes in the graph.
    - edges: List of edges in the graph represented as tuples (tail, head).

    Returns:
    - csr_matrix: Sparse incidence matrix.
    """
    
    # Dimension of the incidence matrix
    dim = len(nodes), len(edges)
    
    # Dictionary to map nodes to their indices
    node_dict = {node: i for i, node in enumerate(nodes)}
    
    # Lists to store row, column, and data of the matrix
    row, col, data = [], [], []
    
    # Iterate over edges to populate the matrix
    for k, (tail, head) in enumerate(edges):
        
        # Index of the tail node
        i = node_dict[tail]
        
        # Index of the head node
        j = node_dict[head]
        
        # Add entries for both directions (tail to head and head to tail)
        row.extend([i, j])
        col.extend([k, k])
        data.extend([1, -1])
        
    # Create a sparse matrix using the collected data
    return csr_matrix((data, (row, col)), shape=dim)


def pairify(items: Iterable, mode: str = 'linear') -> List:
    """
    Converts a group of items into a list of pairs of neighboring items.

    Parameters:
    - items: Iterable of items to be paired.
    - mode: String indicating the pairing mode. 'linear' for linear pairs, 'cyclic' for pairs with wrap-around.

    Returns:
    - List: List of pairs of neighboring items.
    """
    
    # Linear pairing mode
    if mode == 'linear':
        
        # Generate pairs of neighboring items
        i = items[:-1]  # All items except the last one
        j = items[1:]   # All items except the first one
        
    # Cyclic pairing mode
    elif mode == 'cyclic':
        
        # Convert items into a list (to ensure indexing)
        i = list(items)
        j = list(items)
        
        # Move the first item to the end to create a wrap-around
        j.append(j.pop(0))
        
    # Zip the paired items to create pairs
    pairs_list = list(zip(i, j))
    return pairs_list


def cir_mat(edges: List[Tuple[Node, Node]]) -> csr_matrix:
    """
    Returns the circuit matrix (loop-edge) with dimension: e-n+1 x e.

    Parameters:
    - edges: List of edges in the graph represented as tuples (tail, head).

    Returns:
    - csr_matrix: Sparse circuit matrix.
    """
    
    # Find all cycles in the graph using NetworkX
    loops = nx.cycle_basis(nx.Graph(edges))
    
    # Dimension of the circuit matrix
    dim = len(loops), len(edges)
    
    # Dictionary to map edges to their indices and directions
    edgedict = {}
    for i, (tail, head) in enumerate(edges):
        edgedict[tail, head] = (i, +1)  # Positive direction
        edgedict[head, tail] = (i, -1)  # Negative direction
        
    # Lists to store row, column, and data of the matrix
    row, col, data = [], [], []
    
    # Iterate over loops to populate the matrix
    for i, loop in enumerate(loops):
        for pair in pairify(loop, 'cyclic'):
            j, val = edgedict[pair]
            row.append(i)
            col.append(j)
            data.append(val)
            
    # Create a sparse matrix using the collected data
    return csr_matrix((data, (row, col)), shape=dim)

def draw(graph: Graph, ax: plt.Axes = None, 
        pos: Dict = None, directed: bool = True, 
        nshow: bool = True, nlbl: str = 'key', ndecs: int = None, 
        eshow: bool = True, elbl: str = None, edecs: int = None) -> None:
    """
    Pops-up a quick view of the network and its properties at choice.
    This is a 2D view regardless of the object dimensions. Any further 
    rendering should be handled by dedicated applications according to
    the specific needs of each model, or use a third-party software, such
    as Gephi. For 3D objects, the user can rotate the coordinates if wished.

    Parameters:
    - graph: Graph object representing the network.
    - ax: Optional matplotlib Axes object for plotting.
    - pos: Optional dictionary specifying node positions for the graph layout.
    - directed: Boolean indicating whether the graph is directed.
    - nshow: Boolean indicating whether to show nodes.
    - nlbl: String specifying the node attribute to use as labels.
    - ndecs: Optional integer specifying the number of decimal places to round node labels.
    - eshow: Boolean indicating whether to show edges.
    - elbl: String specifying the edge attribute to use as labels.
    - edecs: Optional integer specifying the number of decimal places to round edge labels.

    Returns:
    - None
    """
    
    # Create a NetworkX graph object
    G = nx.Graph()
    G.add_nodes_from(graph.nodes)
    G.add_edges_from(graph.edges)

    # Convert the graph to undirected if required
    if not directed:
        G = G.to_undirected()

    # Generate node positions if not provided
    if not pos:
        pos = nx.layout.kamada_kawai_layout(G)

    # Create a new figure and axis if not provided
    if not ax:
        fig, ax = plt.subplots()
    ax.axis('equal')

    # Draw nodes if required
    if nshow:
        nx.draw_networkx_nodes(G, pos=pos, node_size=150, alpha=0.3, ax=ax)

    # Draw node labels if required
    if nlbl:
        vals = np.array(graph.nodes.get(nlbl))
        if ndecs:
            vals = np.round(vals, ndecs)
        labels = dict(zip(graph.nodes, vals))
        nx.draw_networkx_labels(G, pos=pos, labels=labels, ax=ax)

    # Draw edges if required
    if eshow:
        ecol = None
        if edecs:
            ecol = np.array(graph.edges.get(elbl))
        nx.draw_networkx_edges(G, pos=pos, ax=ax, edge_cmap=plt.cm.magma, edge_color=ecol, node_size=0, width=1)

    # Draw edge labels if required
    if elbl:
        vals = np.array(graph.edges.get(elbl))
        if edecs:
            vals = np.round(vals, edecs)
        labels = dict(zip(graph.edges, vals))
        nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=labels, ax=ax)


if __name__ == '__main__':
    # Example usage
    n0 = Node(key='0')
    n1 = Node(key='1')
    n2 = Node(key='2')
    
    e0 = Edge(n0, n1, key='01')
    e1 = Edge(n1, n2, key='12')
    e2 = Edge(n2, n0, key='20')
    
    g = Graph(Nodes(n0, n1, n2), Edges(e0, e1, e2))

    # Generate adjacency, incidence, and circuit matrices
    adj = adj_mat(g.nodes, g.edges)
    inc = inc_mat(g.nodes, g.edges)
    cir = cir_mat(g.edges) 

    # Draw the graph
    draw(graph=g)
    plt.show()
