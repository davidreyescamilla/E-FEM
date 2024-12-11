import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from scipy.sparse.linalg import spsolve


# Model:

# The model's geometry is expressed in a list containing its nodes. The nodes are described by their coordinates, so we will identify the nodes by its index in this array, and we will bw able to access its coordinates with that index:
nodes = np.array([[0,0],
                  [1,0],
                  [1,1],
                  [0,1]])

# The elements are conformed as per the node's connectivity. We will fins the elements listed in an array, and we will be able to access them by their index. Every item in tis array will contain the nodes in every element:
elements = np.array([[1, 2, 3],
                     [3, 4, 1]])

for i, element in enumerate(elements):
    # Correct indexing for Python (0-based)
    element = element - 1
    elements[i] = element


# Boundary conditions:

# Let's create variables for the traction to easily modify them:
surface_traction = 1000  # Presure in N/m^2

# Now, we establish the boundary conditions: [segment coordinates [init, end], type, value (scalar (value) or vector (x,y))]
# BC type legend:
#   U = displacements (m)
#   f = distributed force (Pa)

boundary_conditions = [[[[0,0],[0,1]],'U',0,"free"], [[[0,0],[1,0]],'U',"free",0], [[[0,1],[1,1]],'U',"free",0], [[[1,0],[1,1]],'f',surface_traction,0]]
    # Note: for displacements we can use "free"


# Material properties

E = 26e9        # Young's modulus (Pa)
nu = 0.2        # Poisson's ratio
sigma_y = 2e6   # Tensile strenght (Pa)
G_f = 74        # Fracture energy (N/m)


# Plot the geometry
import matplotlib.pyplot as plt

def plot_mesh(nodes, elements, boundary_conditions):
    for element in elements:
        polygon = [nodes[node - 1] for node in element]
        polygon.append(polygon[0])  # Close the loop
        plt.plot(*zip(*polygon), marker='o', color='blue')

    for segment, bc_type, _, _ in boundary_conditions:
        plt.plot(*zip(*segment), color='red' if bc_type == 'U' else 'green')

    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

plot_mesh(nodes, elements, boundary_conditions)


# FEM:

# Then, we will create a function to compute the area (to know the area of the element, det(J) = 2A). For a given "element" (listed in the array "elements") we will compute their area by accessing to their nodal coordinates, listed in the array "nodes":
def area(element):
    """Compute the area of an element"""
    x1, y1 = nodes[element[0]]
    x2, y2 = nodes[element[1]]
    x3, y3 = nodes[element[2]]
    return abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2)

# The following function will help with automatizing the computation of the shape functions. This simply changes the position of the "nodes" in an "element":
def cyclic_permutation(element):
    """Perform cyclic permutation of an element's nodes"""
    return np.array([element[1], element[2], element[0]])


def N(element, x, y):
    """Compute the [N] matrix"""
    N = np.zeros([2, 2*len(element)])

    for permutation in range(len(element)): # For every permutation we compute the shape functions corresponding to one node (indices [0, 1, 2] in the permuted "element" array:
        x1, y1 = nodes[element[0]]
        x2, y2 = nodes[element[1]]
        x3, y3 = nodes[element[2]]
        ai = x2 * y3 - x3 * y2
        bi = y2 - y3
        ci = x3 - x2
        shape_function = (ai + bi * x + ci * y)
        N[0][2*permutation] = shape_function
        N[1][2*permutation+1] = shape_function

        element = cyclic_permutation(element)

    return N / 2 * area(element)


def B(element): # The [B] matrix for a CST element is constant, therefore it is independent from x or y.
    """Compute the [B] matrix of an element"""
    B = np.zeros([3, 2 * len(element)])

    for permutation in range(len(element)):
        x1, y1 = nodes[element[0]]
        x2, y2 = nodes[element[1]]
        x3, y3 = nodes[element[2]]

        B[0, 2*permutation] = y2 - y3
        B[1, 2*permutation+1] = x3 - x2
        B[2, 2*permutation] = x3 - x2
        B[2, 2*permutation+1] = y2 - y3

        element = cyclic_permutation(element)

    return B / 2 * area(element)


# Define a function to compute the material constitutive matrix. This will deppend on the type of problem treated; in this case, plane strain for an isotropic material is assumed:
def C(E, nu):
    """Compute the material constitutive law matrix"""
    C = np.array([[1-nu, nu, 0],
                  [nu, 1-nu, 0],
                  [0, 0, (1-2*nu)/2]]) # CHECK: /2 correction for Voight notation strain tensor.

    return C * E/((1+nu)*(1-2*nu))


# Functions to check boundary conditions and applied surface tractions:
# We define a function that checks if a given node is in a certain segment:
def node_in_segment(node, segment):
    """Check if a node is in a segment"""
    v1 = nodes[int(node)] - np.array(segment[1])
    v2 = np.array(segment[0]) - np.array(segment[1])

    if np.cross(v1,v2) == 0 and \
       (min(segment[0][0],segment[1][0]) <= nodes[node][0] <= max(segment[0][0],segment[1][0])) and \
       (min(segment[0][1],segment[1][1]) <= nodes[node][1] <= max(segment[0][1],segment[1][1])):
        return True
    else:
        return False

# Assembly of the global stiffness matrix and force vector:
def assembly(elements):
    """Assemble the global stiffness matrix and force vector"""
    # Global stiffness matrix initialization
    n_dofs = 2 * len(nodes)
    K = np.zeros([n_dofs, n_dofs])

    # Global force vector initialization:
    F = np.zeros(n_dofs)

    # Vectorized assembly
    element_dofs = []
    values = []

    C_mat = C(E, nu)

    for element in elements:
        # Compute local stiffness matrix. This is int_{\Omega} [B]^T [C] [B] d\Omega.
        B_mat = B(element)
        K_e = area(element) * (B_mat.T @ C_mat @ B_mat) # Numerical integration is performed by multiplying bt the area, as both [B] and [C] are independent from x and y

        # Get global DOF indices for the current element
        global_dofs = np.array([2 * node + i for node in element for i in range(2)]) # For every element, this creates a list of the position in the global stiffness matrix of the dofs "i" of every "node"

        # Record contributions to the global matrix
        for i in range(2 * len(element)):
            for j in range(2 * len(element)):
                element_dofs.append((global_dofs[i], global_dofs[j])) # Record the global DOFs in a specific "element"
                values.append(K_e[i, j]) # Record the "K_e" value to add in the global "K" matrix

        # Global force vector assembly:
        # Loop over edges of the element
        for i in range(len(element)):
            # Current edge nodes
            n1, n2 = element[i], element[(i + 1) % len(element)]

            # Check if edge matches a boundary condition
            for condition in boundary_conditions:
                segment, bc_type, tx, ty = condition

                if bc_type == 'f' and node_in_segment(n1, segment) and node_in_segment(n2, segment):
                    # Compute edge length
                    edge_length = np.linalg.norm(nodes[n2] - nodes[n1])

                    # Traction vector
                    traction = np.array([tx, ty])

                    # Local force contributions (constant distribution)
                    f_local = (edge_length / 2) * np.array([*traction, *traction])

                    # Add contributions to the global force vector
                    global_dofs = [2 * n1, 2 * n1 + 1, 2 * n2, 2 * n2 + 1]
                    for dof, f in zip(global_dofs, f_local):
                        F[dof] += f

    # Convert to sparse matrix format
    from scipy.sparse import coo_matrix

    row_indices, col_indices = zip(*element_dofs)
    K_sparse = coo_matrix((values, (row_indices, col_indices)), shape=(n_dofs, n_dofs))
    K = K_sparse.toarray()  # Convert back to dense if needed

    # Impose boundary conditions on displacement:
    for condition in boundary_conditions:
        segment, bc_type, ux, uy = condition
        if bc_type == 'U':
            for i, node in enumerate(nodes):
                if node_in_segment(i, segment):
                    if ux != "free":
                        F[2 * i] = ux
                        K[:, 2 * i] = 0
                        K[2 * i, :] = 0
                        K[2 * i, 2 * i] = 1
                    if uy != "free":
                        F[2 * i + 1] = uy
                        K[:, 2 * i + 1] = 0
                        K[2 * i + 1, :] = 0
                        K[2 * i + 1, 2 * i + 1] = 1
    return K, F

# Solve for displacements
K, F = assembly(elements)
u = spsolve(K, F)
print(u.reshape(-1, 2))

epsilon = np.zeros([len(elements), 3])
sigma = np.zeros([len(elements), 3])
sigma_von_mises = np.zeros(len(elements))

# Solve for strain and stress
for i, element in enumerate(elements):
    # Correct indexing for Python (0-based)
    element = element - 1
    u_e = np.zeros(2 * len(element))
    for j, node in enumerate(element):
        u_e[2 * j] = u[2 * node]
        u_e[2 * j + 1] = u[2 * node + 1]
    epsilon[i] = B(element) @ u_e
    sigma[i] = C(E, nu) @ epsilon[i]

# Plot the displacements
# Create the triangulation object
triangulation = tri.Triangulation(nodes[:, 0], nodes[:, 1], elements)

# Compute magnitude of displacements for visualization
u_x = u[0::2]  # x-displacement (even indices)
u_y = u[1::2]  # y-displacement (odd indices)
u_magnitude = np.sqrt(u_x**2 + u_y**2)

# Plot the displacement magnitude
plt.figure(figsize=(8, 6))
plt.tricontourf(triangulation, u_magnitude, levels=20, cmap='viridis')
plt.colorbar(label='Displacement Magnitude (m)')
plt.title('Displacement Magnitude')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.gca().set_aspect('equal', adjustable='box')
plt.show()


# E-FEM considering strong discontinuities

"""
Linear system of equations:
[K_bb  K_bw  K_bs       ]   {\delta d              }   {- A{f_int^e - f_ext^e}}
[K_wb  K_ww  K_ws       ] · {\delta [|\varepsilon|]} = {- h_[|\varepsilon|]   }
[K_s*b K_s*w K_s*s + K_q]   {\delta [u]            }   {- \Phi_0              }

Simplification to account only for strong discontinuities:
[K_bb  K_bs       ]   {\delta d              }   {- A{f_int^e - f_ext^e}}
[K_s*b K_s*s + K_q]   {\delta [u]            } = {- \Phi_0              }

-\Phi_0 = 0 => K_s*b · \delta d + (K_s*s + K_q) · \delta [u] = 0 => \delta [u] = (K_s*s + K_q)^(-1) · K_s*b · \delta d

So the entire system can be reduced to:
K_sc · \delta d = - A{f_int^e - f_ext^e}
Where:
K_sc = (K_bb - K_bs · (K_s*s + K_q)^(-1) · K_s*b)

Localization criteria: In this first iteration, the crack is predefined, so we know beforehand its location and normal vector
"""
from scipy.special import lambertw

crack_opening = 0.5 # x coordinate of the crack opening
n = np.array([1, 0]) # Normal vector of the crack opening

n_e = np.zeros(2 * len(element))
n_e[0::2], n_e[1::2] = n[0], n[1] # We populate a vector to match the dimension of the expanded matrices

def heaviside(x):
    """Compute the heaviside function"""
    if x < crack_opening: return 0
    # elif node == crack_opening: return 0.5 # This line can be problematic if a node of an element is in the crack opening
    else: return 1

def G_s(element):
    """Compute the G_s matrix"""
    x1, y1 = nodes[element[0]]
    x2, y2 = nodes[element[1]]
    x3, y3 = nodes[element[2]]
    b1, b2, b3 = (y2 - y3) * heaviside(x1), (y3 - y1) * heaviside(x2), (y1 - y2) * heaviside(x3)
    c1, c2, c3 = (x3 - x2) * heaviside(x1), (x1 - x3) * heaviside(x2), (x2 - x1) * heaviside(x3)
    # In the E-FEM formulation, we find a sum for all elements in \Omega^+, however for the SKON formulation G = dH, so by the moment G will be defined at an elementary level and then assembled in a global matrix
    dphi_dx = (b1 + b2 + b3) / (2 * area(element))
    dphi_dy = (c1 + c2 + c3) / (2 * area(element))
    # Compute \nabla \phi:
    G_s = np.zeros((3, 2 * len(element)))
    G_s[0, 0::2] = dphi_dx
    G_s[1, 1::2] = dphi_dy
    G_s[2, 0::2] = dphi_dy
    G_s[2, 1::2] = dphi_dx
    return G_s

def H_s(n):
    """Compute the H_s matrix"""
    nx, ny = n
    H_s = np.zeros((3, 2 * len(element)))
    H_s[0, 0::2] = nx
    H_s[1, 1::2] = ny
    H_s[2, 0::2] = ny
    H_s[2, 1::2] = nx
    return H_s

def K_bb(element):
    """Compute the stiffness matrix K_bb"""
    B_mat = B(element)
    C_mat = C(E, nu)
    return area(element) * (B_mat.T @ C_mat @ B_mat)

def K_bs(element, n):
    """Compute the stiffness matrix K_bs"""
    B_mat = B(element)
    C_mat = C(E, nu)
    G_s_mat = G_s(element)
    return area(element) * (B_mat.T @ C_mat @ G_s_mat @ n)

def K_sb(element, n):
    """Compute the stiffness matrix K_sb"""
    H_s_mat = H_s(n)
    C_mat = C(E, nu)
    B_mat = B(element)
    return n.T @ H_s_mat.T @ C_mat @ B_mat

def K_ss(element, n):
    """Compute the stiffness matrix K_ss"""
    H_s_mat = H_s(n)
    C_mat = C(E, nu)
    G_s_mat = G_s(element)
    return n.T @ H_s_mat.T @ C_mat @ G_s_mat @ n # According to the notes it should be 1 / area(element) * (n.T @ H_s_mat.T @ C_mat @ G_s_mat @ n), but that is not dimensionally consistent

def K_q(sigma_y, G_f, u_modulus):
    """Compute the stiffness matrix K_q"""
    return sigma_y**2 / G_f * np.exp(- sigma_y / G_f * u_modulus)

def u_magnitude(sigma_y, G_f, d, element, n):
    """Compute the magnitude of the crack displacement"""
    T_e = K_sb(element, n) @ d
    M = K_ss(element, n)
    return G_f / sigma_y * (lambertw(sigma_y**2 * np.exp(sigma_y * T_e / (G_f * M) / (G_f * M))) - sigma_y * T_e / (G_f * M))

def K_sc(element, n):
    """Compute the stiffness matrix K_sc"""
    K_bb_mat = K_bb(element)
    K_bs_mat = K_bs(element, n)
    K_sb_mat = K_sb(element, n)
    K_ss_mat = K_ss(element, n)
    return K_bb_mat - K_bs @ np.linalg.inv(K_ss_mat + K_q) @ K_sb


# Localization and opening criteria:
"""
With this first (purely geometric) approach, we will project the traction vector in the normal direction to the crack opening (predefined).
Localization will take place when this component of the traction vector reaches a treshold (sigma_y)
"""
def T(element, n, sigma_e):
    """Compute the traction vector for a give element"""
    nx, ny = n
    H_s_mat = np.zeros((len(element), 2)) # Notice it is more convinient to redefine H_s for this specific case
    H_s_mat[0, 0] = nx
    H_s_mat[1, 1] = ny
    H_s_mat[2, 0] = ny
    H_s_mat[2, 1] = nx
    return H_s_mat.T @ sigma_e


# Resolution of the system
"""
In this case, as we are no longer facing a linear problem, we need a slightly different approach.
We will then implement an iterative solver with a gradual increment of the load.

BFGS algorithm: https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm

Implementation of BFGS in EFEM
1. Initialization:
  - Start with an initial guess for the displacement field.
  - Set an initial stiffness matrix approximation (usually the identity matrix or a tangent stiffness matrix).
2. Iterative Process:
  - Compute the residual:
    R = K·Δu − F
    Where K is the stiffness matrix, Δu is the displacement increment, and F is the external force vector.
  - Update the displacement increment Δu using:
    Δu = −H^(-1)·R
    Where: H is the current approximation of the stiffness matrix.
  - Update H using the BFGS formula:
    H_k+1 = H_k + y_k·y_k^T / (y_k^T·s_k) - H_k·s_k·s_k^T·H_k / (s_k^T·H_k·s_k)
    Here:
    s_k = u_k+1 - u_k
    y_k = R_k+1 - R_k
3. Crack Handling:
  - Modify the stiffness matrix locally for elements crossed by a discontinuity. This will affect the residual and H, requiring careful implementation.
  - Update the displacement increment and repeat the BFGS iterations until convergence.
4. Convergence Criteria:
  - Check for convergence using a norm of the residual or displacement increments. A common criterion is: ∥R∥ < ϵ
    where ϵ is a small tolerance value.
"""

# Assembly of the global stiffness matrix and force vector:

# Global stiffness matrix initialization
n_dofs = 2 * len(nodes)
K = np.zeros([n_dofs, n_dofs])

# Global force vector initialization:
F = np.zeros(n_dofs)

# Vectorized assembly
element_dofs = []
values = []

C_mat = C(E, nu)

for element in elements:
    # Compute local stiffness matrix. This is int_{\Omega} [B]^T [C] [B] d\Omega.
    B_mat = B(element)
    K_e = area(element) * (B_mat.T @ C_mat @ B_mat) # Numerical integration is performed by multiplying bt the area, as both [B] and [C] are independent from x and y

    # Get global DOF indices for the current element
    global_dofs = np.array([2 * node + i for node in element for i in range(2)]) # For every element, this creates a list of the position in the global stiffness matrix of the dofs "i" of every "node"

    # Record contributions to the global matrix
    for i in range(2 * len(element)):
        for j in range(2 * len(element)):
            element_dofs.append((global_dofs[i], global_dofs[j])) # Record the global DOFs in a specific "element"
            values.append(K_e[i, j]) # Record the "K_e" value to add in the global "K" matrix

    # Global force vector assembly:
    # Loop over edges of the element
    for i in range(len(element)):
        # Current edge nodes
        n1, n2 = element[i], element[(i + 1) % len(element)]

        # Check if edge matches a boundary condition
        for condition in boundary_conditions:
            segment, bc_type, tx, ty = condition

            if bc_type == 'f' and node_in_segment(n1, segment) and node_in_segment(n2, segment):
                # Compute edge length
                edge_length = np.linalg.norm(nodes[n2] - nodes[n1])

                # Traction vector
                traction = np.array([tx, ty])

                # Local force contributions (constant distribution)
                f_local = (edge_length / 2) * np.array([*traction, *traction])

                # Add contributions to the global force vector
                global_dofs = [2 * n1, 2 * n1 + 1, 2 * n2, 2 * n2 + 1]
                for dof, f in zip(global_dofs, f_local):
                    F[dof] += f

# Convert to sparse matrix format
from scipy.sparse import coo_matrix

row_indices, col_indices = zip(*element_dofs)
K_sparse = coo_matrix((values, (row_indices, col_indices)), shape=(n_dofs, n_dofs))
K = K_sparse.toarray()  # Convert back to dense if needed

# Impose boundary conditions on displacement:
for condition in boundary_conditions:
    segment, bc_type, ux, uy = condition
    if bc_type == 'U':
        for i, node in enumerate(nodes):
            if node_in_segment(i, segment):
                if ux != "free":
                    F[2 * i] = ux
                    K[:, 2 * i] = 0
                    K[2 * i, :] = 0
                    K[2 * i, 2 * i] = 1
                if uy != "free":
                    F[2 * i + 1] = uy
                    K[:, 2 * i + 1] = 0
                    K[2 * i + 1, :] = 0
                    K[2 * i + 1, 2 * i + 1] = 1
