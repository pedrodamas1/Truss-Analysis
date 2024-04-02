import numpy as np
from scipy.linalg import solve
from graph import Node, Edge, Nodes, Edges, Graph
from analysis import inc_mat, cir_mat


class Joint(Node):
	"""Joint class represents a junction in the circuit."""
	
	def __init__(self, **kwargs) -> None:
		"""Initialize a Joint object."""
		super().__init__(**kwargs)
		self.v = 0


class Source(Edge):
	"""Source class represents a voltage source in the circuit."""
	
	def __init__(self, tail: Joint, head: Joint, v: float, **kwargs) -> None:
		"""
		Initialize a Source object.

		Parameters:
		tail -- the tail node of the edge
		head -- the head node of the edge
		v -- voltage of the source
		"""
		super().__init__(tail, head, **kwargs)
		self.r = 0
		self.i = 0
		self.v = v


class Resistor(Edge):
	"""Resistor class represents a resistor in the circuit."""
	
	def __init__(self, tail: Joint, head: Joint, r: float, **kwargs) -> None:
		"""
		Initialize a Resistor object.

		Parameters:
		tail -- the tail node of the edge
		head -- the head node of the edge
		r -- resistance value of the resistor
		"""
		super().__init__(tail, head, **kwargs)
		self.r = r
		self.i = 0
		self.v = 0
		

class Circuit(Graph):
	"""Circuit class represents an electrical circuit."""
	
	def solve(self) -> None:
		"""Solve the circuit and update edge currents and node voltages."""
		
		# Get the circuit matrix of the system
		B = cir_mat(self.edges)
		
		# Get the resistance of each edge
		r = self.edges.get('r')
		
		# Get the voltage of each edge
		v = self.edges.get('v')
		
		# Calculate the Laplacian Matrix of the mesh
		R = B @ np.diag(r) @ B.T
		
		# Calculate the loop voltage
		vl = B @ v
		
		# Try to solve the system of linear equations using Gaussian Elimination - finding loop currents
		try:
			il = solve(R, vl)
		except np.linalg.LinAlgError:
			raise ValueError("Singular matrix: Circuit may have unresolved loops or short circuits.")
		
		# Calculate the edge currents using the loop currents
		i = B.T @ il
		
		# Set the final values
		self.edges.set('i', i)
		
		# Get the incidence matrix
		A = inc_mat(self.nodes, self.edges)
		
		# Use the Incidence Matrix and Least Squares to calculate the nodal voltages
		try:
			v = np.linalg.lstsq(-A.toarray().T, v - r * i, rcond=None)[0]
		except np.linalg.LinAlgError:
			raise ValueError("Singular matrix: Circuit may have unresolved loops or short circuits.")
		
		# Set the nodal voltage values
		self.nodes.set('v', v - np.min(v))
		
		return None

	def residuals(self) -> float:
		"""Returns the residuals of the solution"""

		# Validate result with kirchhoff current law (KCL) and kirchhoff voltage law (KVL)
		r = self.edges.get('r') # branch resistance
		i = self.edges.get('i') # branch current
		v = self.edges.get('v') # branch voltage

		# Get topological matrices
		A = inc_mat(self.nodes, self.edges)
		B = cir_mat(self.edges)

		# Calculate residuals
		F = np.concatenate((A@i, B@(r*i-v)))
		return np.linalg.norm(F)


if __name__ == '__main__':
	n0 = Joint()
	n1 = Joint()
	n2 = Joint()
	n3 = Joint()
	
	e0 = Source(n0, n1, 10)
	e1 = Resistor(n1, n2, 10)
	e2 = Resistor(n2, n3, 20)
	e3 = Source(n3, n0, -20)
	e4 = Resistor(n2, n0, 40)
	
	c = Circuit(Nodes(n0, n1, n2, n3), Edges(e0, e1, e2, e3, e4))
	c.solve()

	print(c.edges.get('i'))
	print(c.residuals())
