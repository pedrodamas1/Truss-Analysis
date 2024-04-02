from graph import Node, Edge, Nodes, Edges, Graph
from typing import Tuple, List
from scipy import sparse
from scipy.linalg import solve
import numpy as np
np.set_printoptions(linewidth=np.inf)


class Joint(Node):
	'''Class to represent a joint in a 2D truss'''
	def __init__(self, pos, fext, disp, dof, **kwargs) -> None:
		self.pos = pos # position
		self.fext = fext # external forces
		self.disp = disp # displacement
		self.dof = dof # degrees of freedom
		super().__init__(**kwargs)


class Member(Edge):
	'''Class to represent a joint in a truss'''
	def __init__(self, tail: Joint, head: Joint, young: float, area: float, **kwargs) -> None:
		super().__init__(tail, head, **kwargs)
		self.young = young # young's modulus
		self.area = area # xsection area

	def get_length(self) -> float:
		"""Member length"""
		return np.linalg.norm(self.head.pos - self.tail.pos)

	def get_angle(self) -> float:
		"""Member angle"""
		x,y = self.head.pos - self.tail.pos
		return np.arctan2(y, x)
	
	def get_transformation_matrix(self):
		"""Transformation matrix"""
		return np.array([
			[
				np.cos(self.get_angle()), 
				np.sin(self.get_angle()), 
				0., 
				0.
			], 
			[
				0., 
				0., 
				np.cos(self.get_angle()), 
				np.sin(self.get_angle())
			]
		])

	def get_local_member_stiffness_matrix(self):
		"""Local member stiffness matrix Kl"""
		return self.young*self.area/self.get_length() * np.array([[1, -1],[-1, 1]])
	
	def get_global_member_stiffness_matrix(self):
		"""Global member stiffness matrix Kg"""
		return self.get_transformation_matrix().T @ self.get_local_member_stiffness_matrix() @ self.get_transformation_matrix()
	
	def get_Kp(self, N:int, indices:List[int]):
		"""Returns the single member stiffness matrix"""
		row, col, data = [], [], []
		Kg = self.get_global_member_stiffness_matrix()
		for i, line in enumerate(Kg):
			for j, item in enumerate(line):
				row.append( indices[i] )
				col.append( indices[j] )
				data.append(item)
		# for i in range(4):
		# 	for j in range(4):
		# 		row.append( indices[i] )
		# 		col.append( indices[j] )
		# 		data.append(Kg[i,j])
		Kp = sparse.csr_array((data, (row, col)), shape=(N,N))
		return Kp

	@property
	def u(self):
		return np.vstack((self.tail.disp, self.head.disp))

	@property
	def f(self) -> float:
		"""Returns the member force"""
		u = self.u.flatten()
		u1, u2 = self.get_transformation_matrix() @ u
		return self.young*self.area/self.get_length() * (u2-u1)


class Truss(Graph):
	"""Class to represent a truss structure"""
	
	@property
	def K(self):
		"""Return the global stiffness matrix"""

		# Fetch all keys
		keys = self.nodes.get('key')

		# Get the number of DOFS
		N = 2*len(keys)
		
		# Create an empty matrix for the stiffness matrix
		K = np.zeros((N,N))

		# Create a nodes dictionary for the order indexing
		nodes_dict = {key: i for i, key in enumerate(keys)}

		# Loops over the members to get their stiffness matrices
		member : Member
		for member in self.edges:
			i,j = member.tail.key, member.head.key
			k = nodes_dict[i]
			l = nodes_dict[j]
			K += member.get_Kp(N, [2*k, 2*k+1, 2*l, 2*l+1])

		return K

	def solve(self) -> None:
		"""solves the stiffness matrix system"""
		
		# Get the degrees of freedom and external forces. Use np.ravel([A,B], 'F) to concatenate alternating
		dof = self.nodes.get('dof').flatten()
		fext = self.nodes.get('fext').flatten()

		# Get the indices of free nodes
		indices = np.where(dof == 1)[0]

		# Get the global truss stiffness matrix
		K = self.K

		# Calculate the displacements of free DOFs
		utemp = solve( K[indices,:][:,indices], fext[indices] ) # ax = b

		# Get the number of DOFS. For 3D trusses, the dimension is (N,3)
		N = len(self.nodes)
		dim = (N,2)

		# Reconstruct and set the calculated displacements. For 3D is 3*N
		u = np.zeros(2*N)
		np.put(u, indices, utemp)
		self.nodes.set('disp', np.reshape(u, dim))

		# Calculate the reaction forces
		self.nodes.set('fext', np.reshape(K @ u, dim))



if __name__ == '__main__':
	j0 = Joint(pos=np.array([0,0]), fext=np.array([0,0]), disp=np.array([0,0]), dof=np.array([0,0]), key=0)
	j1 = Joint(pos=np.array([4,0]), fext=np.array([0,0]), disp=np.array([0,0]), dof=np.array([0,0]), key=1)
	j2 = Joint(pos=np.array([8,0]), fext=np.array([0,0]), disp=np.array([0,0]), dof=np.array([0,0]), key=2)
	j3 = Joint(pos=np.array([4,-6]), fext=np.array([1.e5, -1.e5]), disp=np.array([0,0]), dof=np.array([1,1]), key=3)

	m0 = Member(tail=j0, head=j3, young=2.e11, area=5.e-3)
	m1 = Member(tail=j1, head=j3, young=2.e11, area=5.e-3)
	m2 = Member(tail=j2, head=j3, young=2.e11, area=5.e-3)

	t = Truss(
		nodes=Nodes(j0, j1, j2, j3),
		edges=Edges(m0, m1, m2)
	)

	t.solve()

	from analysis import draw
	import matplotlib.pyplot as plt

	# Get the nodal position
	pos = dict(zip(t.nodes, t.nodes.get('pos')))

	# Create just a figure and only one subplot
	fig, ax = plt.subplots()
	ax.set_title('Title')

	# Draw a simple plot of the truss
	draw(graph=t, ax=ax, pos=pos, nlbl='key', eshow=True, elbl=None)

	# Get the nodal position
	pos = dict(zip(t.nodes, t.nodes.get('pos')+1000*t.nodes.get('disp')))

	# Draw a simple plot of the truss
	draw(graph=t, ax=ax, pos=pos, nlbl='key', eshow=True, elbl='f', edecs=2)
	plt.show()




# """
# https://www.degreetutors.com/direct-stiffness-method/
# """

# from numpy.typing import NDArray



	# @property
	# def coords(self):
	# 	return np.vstack((self.tail.pos, self.head.pos))
	
