from graph import Node, Edge, Nodes, Edges, Graph
from scipy.linalg import solve
import numpy as np
np.set_printoptions(linewidth=np.inf)


class Joint(Node):
	'''
	Class to represent a joint in a 2D truss

	Attributes:
		coordinates (np.ndarray): The coordinates of the joint.
		external_forces (np.ndarray): External forces applied on the joint.
		reaction_forces (np.ndarray): Reaction forces at the joint.
		displacement (np.ndarray): Displacement of the joint.
		degrees_of_freedom (np.ndarray): Degrees of freedom of the joint.
	'''

	def __init__(self, 
			  coordinates: np.ndarray, external_forces: np.ndarray = np.zeros(2), 
			  reaction_forces: np.ndarray = np.zeros(2), displacement: np.ndarray = np.zeros(2), 
			  degrees_of_freedom: np.ndarray = np.zeros(2), **kwargs) -> None:
		'''
		Initialize a Joint object.

		Parameters:
		coordinates (np.ndarray): The coordinates of the joint.
		external_forces (np.ndarray): External forces applied on the joint.
		reaction_forces (np.ndarray): Reaction forces at the joint.
		displacement (np.ndarray): Displacement of the joint.
		degrees_of_freedom (np.ndarray): Degrees of freedom of the joint.
		**kwargs: Additional keyword arguments.
		'''
		self.coordinates = coordinates
		self.external_forces = external_forces
		self.reaction_forces = reaction_forces
		self.displacement = displacement
		self.degrees_of_freedom = degrees_of_freedom
		super().__init__(**kwargs)

	def get_displaced_coordinates(self, scale: float) -> np.ndarray:
			'''
			Get the coordinates of the joint after displacement.

			Parameters:
			scale (float): Scaling factor for displacement.

			Returns:
			np.ndarray: Displaced coordinates.
			'''
			return self.coordinates + scale * self.displacement


class Member(Edge):
	'''
	Class to represent a member in a truss.

	Attributes:
		tail (Joint): The joint at the tail of the member.
		head (Joint): The joint at the head of the member.
		youngs_modulus (float): Young's modulus of the member material.
		area (float): Cross-sectional area of the member.
	'''

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
		'''
		Calculate the length of the member.

		Returns:
		float: Length of the member.
		'''
		length = np.linalg.norm(self.get_displacement_vector())
		return length

	def get_displacement_vector(self) -> np.ndarray:
		'''
		Calculate the displacement vector of the member.

		Returns:
		np.ndarray: Displacement vector.
		'''
		return self.head.coordinates - self.tail.coordinates

	def get_angle(self) -> float:
		'''
		Calculate the angle of the member.

		Returns:
		float: Angle of the member.
		'''
		x, y = self.get_displacement_vector()
		angle = np.arctan2(y, x)
		return angle
	
	def get_transformation_matrix(self) -> np.ndarray:
		'''
		Calculate the transformation matrix.

		Returns:
		np.ndarray: Transformation matrix.
		'''
		angle = self.get_angle()
		c = np.cos(angle)
		s = np.sin(angle)
		transformation_matrix = np.array([
			[c, s, 0, 0],
			[0, 0, c, s]
		])
		return transformation_matrix
	
	def get_local_member_stiffness_matrix(self) -> np.ndarray:
		'''
		Calculate the local member stiffness matrix.

		Returns:
		np.ndarray: Local member stiffness matrix.
		'''
		E = self.youngs_modulus
		A = self.area
		L = self.get_length()
		local_member_stiffness_matrix = (E * A / L) * np.array([[1, -1], [-1, 1]])
		return local_member_stiffness_matrix

	def get_global_member_stiffness_matrix(self) -> np.ndarray:
		'''
		Calculate the global member stiffness matrix.

		Returns:
		np.ndarray: Global member stiffness matrix.
		'''
		T = self.get_transformation_matrix()
		KL = self.get_local_member_stiffness_matrix()
		global_member_stiffness_matrix = T.T @ KL @ T
		return global_member_stiffness_matrix

	@property
	def member_force(self) -> float:
		'''
		Calculate the member force.

		Returns:
		float: Member force.
		'''
		u = np.vstack((self.tail.displacement, self.head.displacement)).flatten()
		T = self.get_transformation_matrix()
		u1, u2 = T @ u
		E = self.youngs_modulus
		A = self.area
		L = self.get_length()
		return E * A / L * (u2 - u1)
	

class Truss(Graph):
	"""Class to represent a truss structure"""
	
	def get_primary_structure_stiffness_matrix(self):
		'''
		Calculate the primary stiffness matrix.

		Returns:
		np.ndarray: Primary stiffness matrix.
		'''
		N = len(self.nodes)
		nodes_dict = dict(zip(self.nodes, range(N)))
		primary_stiffness_matrix = np.zeros((2*N, 2*N))
		member : Member
		for member in self.edges:
			KG = member.get_global_member_stiffness_matrix()
			i = 2*nodes_dict[member.tail]
			j = 2*nodes_dict[member.head]
			primary_stiffness_matrix[i:i+2, i:i+2] += KG[:2, :2]
			primary_stiffness_matrix[i:i+2, j:j+2] += KG[:2, 2:]
			primary_stiffness_matrix[j:j+2, i:i+2] += KG[2:, :2]
			primary_stiffness_matrix[j:j+2, j:j+2] += KG[2:, 2:]
		return primary_stiffness_matrix

	def solve(self) -> None:
		"""Solves the stiffness matrix system"""
		
		# Get the degrees of freedom and external forces
		dof = self.nodes.get('degrees_of_freedom')
		fext = self.nodes.get('external_forces')

		# Get the indices of free nodes flattened out
		idx = np.where(dof.flatten() == 1)[0]

		# Get the global truss stiffness matrix
		K = self.get_primary_structure_stiffness_matrix()

		# Get the structure stiffness matrix by imposing boundary conditions
		structure_stiffness_matrix = K[idx,:][:,idx]

		# Calculate the displacements of free DOFs (Ax = b)
		displacement = self.nodes.get('displacement')
		displacement[np.where(dof == 1)] = solve(structure_stiffness_matrix, fext.flatten()[idx] )
		self.nodes.set('displacement', displacement)

		# Calculate the reaction forces
		reaction_forces = np.reshape(K @ displacement.flatten(), displacement.shape)
		self.nodes.set('reaction_forces', reaction_forces)

		return None


if __name__ == '__main__':
	j0 = Joint(coordinates=np.array([0,0]), key=0)
	j1 = Joint(coordinates=np.array([4,0]), key=1)
	j2 = Joint(coordinates=np.array([8,0]), key=2)
	j3 = Joint(coordinates=np.array([4,-6]), external_forces=np.array([1.e5, -1.e5]), degrees_of_freedom=np.array([1,1]), key=3)

	m0 = Member(tail=j0, head=j3, youngs_modulus=2.e11, area=5.e-3, key='A')
	m1 = Member(tail=j1, head=j3, youngs_modulus=2.e11, area=5.e-3, key='B')
	m2 = Member(tail=j2, head=j3, youngs_modulus=2.e11, area=5.e-3, key='C')

	truss = Truss(
		nodes=Nodes(j0, j1, j2, j3),
		edges=Edges(m0, m1, m2)
	)

	truss.solve()

