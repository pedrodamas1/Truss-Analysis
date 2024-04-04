"""
https://www.degreetutors.com/direct-stiffness-method/
"""

from graph import Node, Edge, Nodes, Edges, Graph
from typing import Tuple, List
from scipy import sparse
from scipy.linalg import solve
import numpy as np
np.set_printoptions(linewidth=np.inf)

# Use this to defined the dimension of the problem
# 2D truss: 2
# 2D frame: 3
# 3D truss: 3
# 3D frame: ??
DIMENSION = 2


class Joint(Node):
	'''Class to represent a joint in a 2D truss'''

	def __init__(self, 
			  coordinates: np.ndarray, 
			  external_forces: np.ndarray = np.zeros(2), 
			  displacement: np.ndarray = np.zeros(2), 
			  degrees_of_freedom: np.ndarray = np.zeros(2), 
			  **kwargs) -> None:
		'''
		Initialize a Joint object.

		Parameters:
		coordinates (np.ndarray): The coordinates of the joint.
		external_forces (np.ndarray): External forces applied on the joint.
		displacement (np.ndarray): Displacement of the joint.
		degrees_of_freedom (np.ndarray): Degrees of freedom of the joint.
		**kwargs: Additional keyword arguments.
		'''
		self.coordinates = coordinates
		self.external_forces = external_forces
		self.displacement = displacement
		self.degrees_of_freedom = degrees_of_freedom
		super().__init__(**kwargs)


class Member(Edge):
	'''Class to represent a member in a truss'''

	def __init__(self, tail: Joint, head: Joint, youngs_modulus: float, area: float, **kwargs) -> None:
		'''
		Initialize a Member object.

		Parameters:
		tail (Joint): The joint at the tail of the member.
		head (Joint): The joint at the head of the member.
		young (float): Young's modulus of the member material.
		area (float): Cross-sectional area of the member.
		**kwargs: Additional keyword arguments.
		'''
		self.tail = tail
		self.head = head
		self.youngs_modulus = youngs_modulus
		self.area = area
		self.__dict__.update(kwargs)

	def get_length(self) -> float:
		"""Calculate the length of the member."""
		length = np.linalg.norm(self.head.coordinates - self.tail.coordinates)
		return length

	def get_angle(self) -> float:
		"""Calculate the angle of the member."""
		x, y = self.head.coordinates - self.tail.coordinates
		angle = np.arctan2(y, x)
		return angle
	
	def get_transformation_matrix(self) -> np.ndarray:
		"""The transformation matrix T acts as a bridge between local and global coordinates"""
		angle = self.get_angle()
		c = np.cos(angle)
		s = np.sin(angle)
		transformation_matrix = np.array([
			[c, s, 0, 0], 
			[0, 0, c, s]
		])
		return transformation_matrix

	def get_local_member_stiffness_matrix(self) -> np.ndarray:
		"""Calculate the local member stiffness matrix."""
		E = self.youngs_modulus
		A = self.area
		L = self.get_length()
		local_member_stiffness_matrix = ( E*A/L) * np.array([[1, -1],[-1, 1]])
		return local_member_stiffness_matrix
	
	def get_global_member_stiffness_matrix(self) -> np.ndarray:
		"""Calculate the global member stiffness matrix."""
		T = self.get_transformation_matrix()
		KL = self.get_local_member_stiffness_matrix()
		global_member_stiffness_matrix = T.T @ KL @ T
		return global_member_stiffness_matrix
	

	# def get_primary_member_stiffness_matrix(self, N:int, indices:List[int]) -> sparse.csr_array:
	# 	"""Primary member Stiffness Matrix KP"""
	# 	row, col, data = [], [], []
	# 	KG = self.get_global_member_stiffness_matrix()
	# 	for i, line in enumerate(KG):
	# 		for j, item in enumerate(line):
	# 			row.append( indices[i] )
	# 			col.append( indices[j] )
	# 			data.append(item)
	# 	primary_member_stiffness_matrix = sparse.csr_array((data, (row, col)), shape=(N,N))
	# 	return primary_member_stiffness_matrix

	# @property
	# def u(self):
	# 	return np.vstack((self.tail.displacement, self.head.displacement))

	# @property
	# def f(self) -> float:
	# 	"""Returns the member force"""
	# 	u = self.u.flatten()
	# 	T = self.get_transformation_matrix()
	# 	u1, u2 = T @ u
	# 	E = self.youngs_modulus
	# 	A = self.area
	# 	L = self.get_length()
	# 	return E*A/L * (u2-u1)


class Truss(Graph):
	"""Class to represent a truss structure"""
	
	def get_primary_stiffness_matrix(self):
	
		nodes = self.nodes
		N = len(nodes)
		nodes_dict = dict(zip(nodes, range(N)))
		primary_stiffness_matrix = np.zeros((2*N, 2*N))
		
		member : Member
		for member in self.edges:
			KG = member.get_global_member_stiffness_matrix()
			K11 = KG[:2, :2]
			K12 = KG[:2, 2:]
			K21 = KG[2:, :2]
			K22 = KG[2:, 2:]
			i = nodes_dict[member.tail]
			j = nodes_dict[member.head]
			primary_stiffness_matrix
			# += member.get_primary_member_stiffness_matrix(N, [2*k, 2*k+1, 2*l, 2*l+1])
			break
	
	# @property
	# def K(self):
	# 	"""Return the global stiffness matrix"""

	# 	# Fetch all keys
	# 	keys = self.nodes.get('key')

	# 	# Get the number of DOFS
	# 	N = 2*len(keys)
		
	# 	# Create an empty matrix for the stiffness matrix
	# 	K = np.zeros((N,N))

	# 	# Create a nodes dictionary for the order indexing
	# 	nodes_dict = {key: i for i, key in enumerate(keys)}

	# 	# Loops over the members to get their stiffness matrices
	# 	member : Member
	# 	for member in self.edges:
	# 		i,j = member.tail.key, member.head.key
	# 		k = nodes_dict[i]
	# 		l = nodes_dict[j]
	# 		K += member.get_primary_member_stiffness_matrix(N, [2*k, 2*k+1, 2*l, 2*l+1])

	# 	return K

	# def solve(self) -> None:
	# 	"""solves the stiffness matrix system"""
		
	# 	# Get the degrees of freedom and external forces. Use np.ravel([A,B], 'F) to concatenate alternating
	# 	degrees_of_freedom = self.nodes.get('degrees_of_freedom').flatten()
	# 	external_forces = self.nodes.get('external_forces').flatten()

	# 	# Get the indices of free nodes
	# 	indices = np.where(degrees_of_freedom == 1)[0]

	# 	# Get the global truss stiffness matrix
	# 	K = self.K

	# 	# Calculate the displacements of free DOFs
	# 	utemp = solve( K[indices,:][:,indices], external_forces[indices] ) # ax = b

	# 	# Get the number of DOFS. For 3D trusses, the dimension is (N,3)
	# 	N = len(self.nodes)
	# 	dim = (N,2)

	# 	# Reconstruct and set the calculated displacements. For 3D is 3*N
	# 	u = np.zeros(2*N)
	# 	np.put(u, indices, utemp)
	# 	self.nodes.set('displacement', np.reshape(u, dim))

	# 	# Calculate the reaction forces
	# 	self.nodes.set('external_forces', np.reshape(K @ u, dim))


if __name__ == '__main__':
	j0 = Joint(coordinates=np.array([0,0]), external_forces=np.zeros(2), displacement=np.zeros(2), degrees_of_freedom=np.zeros(2), key=0)
	j1 = Joint(coordinates=np.array([4,0]), external_forces=np.zeros(2), displacement=np.zeros(2), degrees_of_freedom=np.zeros(2), key=1)
	j2 = Joint(coordinates=np.array([8,0]), external_forces=np.zeros(2), displacement=np.zeros(2), degrees_of_freedom=np.zeros(2), key=2)
	j3 = Joint(coordinates=np.array([4,-6]), external_forces=np.array([1.e5, -1.e5]), displacement=np.zeros(2), degrees_of_freedom=np.array([1,1]), key=3)

	m0 = Member(tail=j0, head=j3, youngs_modulus=2.e11, area=5.e-3, key='A')
	m1 = Member(tail=j1, head=j3, youngs_modulus=2.e11, area=5.e-3, key='B')
	m2 = Member(tail=j2, head=j3, youngs_modulus=2.e11, area=5.e-3, key='C')

	# print((m0.get_global_member_stiffness_matrix()/1e9).round(4))

	t = Truss(
		nodes=Nodes(j0, j1, j2, j3),
		edges=Edges(m0, m1, m2)
	)
	t.get_primary_stiffness_matrix()

	# t.solve()

	# from analysis import draw
	# import matplotlib.pyplot as plt

	# # Get the nodal position
	# pos = dict(zip(t.nodes, t.nodes.get('coordinates')))

	# # Create just a figure and only one subplot
	# fig, ax = plt.subplots()
	# ax.set_title('Title')
	# ax.grid()

	# # Draw a simple plot of the truss
	# draw(graph=t, ax=ax, pos=pos, nlbl='key', eshow=True, elbl=None)

	# # Get the nodal position
	# pos = dict(zip(t.nodes, t.nodes.get('coordinates')+1000*t.nodes.get('displacement')))

	# # Draw a simple plot of the truss
	# draw(graph=t, ax=ax, pos=pos, nlbl='key', eshow=True, elbl='f', edecs=2)
	# plt.show()


