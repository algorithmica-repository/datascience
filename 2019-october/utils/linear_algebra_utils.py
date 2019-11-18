import numpy as np
import math
from matplotlib import ticker
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import cycle

plt.rc('font', family='serif', size=4)
plt.rc('figure', dpi=200)
plt.rc('axes', axisbelow=True, titlesize=6, labelsize=5)
plt.rc('lines', linewidth=1)

grey, gold, lightblue, green, red, darkblue = '#808080', '#cab18c', '#0096d6', '#008367','#E31937', '#004065'
pink, yellow, orange, purple, brown = '#ef7b9d', '#fbd349', '#ffa500', '#a35cff', '#731d1d'
quiver_params = {'angles': 'xy', 'scale_units': 'xy', 'scale': 1, 'width': 0.007}
grid_params = {'linewidth': 0.5, 'alpha': 0.8}
text_params = {'ha': 'center', 'va': 'center', 'size' : 5}

def plot_vector(vectors, tails=None):
    vectors = np.array(vectors)
    assert vectors.shape[1] == 2, "Each vector should have 2 elements."  
    if tails is not None:
        tails = np.array(tails)
        assert tails.shape[1] == 2, "Each tail should have 2 elements."
    else:
        tails = np.zeros_like(vectors)
    
    # tile vectors or tails array if needed
    nvectors = vectors.shape[0]
    ntails = tails.shape[0]
    if nvectors == 1 and ntails > 1:
        vectors = np.tile(vectors, (ntails, 1))
    elif ntails == 1 and nvectors > 1:
        tails = np.tile(tails, (nvectors, 1))
    else:
        assert tails.shape == vectors.shape, "vectors and tail must have a same shape"

    # calculate xlimit & ylimit
    heads = tails + vectors
    limit = np.max(np.abs(np.hstack((tails, heads))))
    limit = np.ceil(limit * 1.2)   # add some margins
    
    figure, axis = plt.subplots()
    
    grid_range = 20
    x = np.arange(-grid_range, grid_range+1)
    X_, Y_ = np.meshgrid(x,x)
    I = np.array([1,0])
    J = np.array([0,1])
    X = I[0]*X_ + J[0]*Y_
    Y = I[1]*X_ + J[1]*Y_    
  
    # draw grid lines
    for i in range(x.size):
        axis.plot(X[i,:], Y[i,:], c=gold, **grid_params)
        axis.plot(X[:,i], Y[:,i], c=lightblue, **grid_params)

    axis.quiver(tails[:,0], tails[:,1], vectors[:,0], vectors[:,1], color=darkblue,  **quiver_params)
    axis.set_xlim([-limit, limit])
    axis.set_ylim([-limit, limit])
    #axis.set_aspect('equal')  
    
    #hide all the spines
    axis.grid(False)
    axis.spines['left'].set_visible(False)
    axis.spines['right'].set_visible(False)
    axis.spines['bottom'].set_visible(False)
    axis.spines['top'].set_visible(False)

def plot_2d_transformation_helper(axis, matrix, vectors, show_values, title=None):
    assert matrix.shape == (2,2), "the input matrix must have a shape of (2,2)"
    grid_range = 20
    x = np.arange(-grid_range, grid_range+1)
    X_, Y_ = np.meshgrid(x,x)
    I = matrix[:,0]
    J = matrix[:,1]
    X = I[0]*X_ + J[0]*Y_
    Y = I[1]*X_ + J[1]*Y_
    origin = np.zeros(1)
    
    # draw grid lines
    for i in range(x.size):
        axis.plot(X[i,:], Y[i,:], c=gold, **grid_params)
        axis.plot(X[:,i], Y[:,i], c=lightblue, **grid_params)
    
    #draw unit vectors
    axis.quiver(origin, origin, [I[0]], [I[1]], color=green, **quiver_params)
    axis.quiver(origin, origin, [J[0]], [J[1]], color=red, **quiver_params)
    if show_values:
        axis.text(I[0]+0.1, I[1]+0.1, '$({},{})$'.format(round(I[0],1), round(I[1],1) ), color=gold, **text_params)
        axis.text(J[0]+0.1, J[1]+0.1, '$({},{})$'.format(round(J[0],1), round(J[1],1) ), color=gold, **text_params)

    # draw vectors
    color_cycle = cycle([pink, darkblue, orange, purple, brown])
    for vector in vectors:
            color = next(color_cycle)
            vector_ = matrix @ vector.reshape(-1,1)
            axis.quiver(origin, origin, [vector_[0]], [vector_[1]], color=color, **quiver_params)
            if show_values:
                axis.text(vector_[0]+0.1, vector_[1]+0.1, '$({},{})$'.format(np.round(vector_[0][0],1), np.round(vector_[1][0],1) ), color=gold, **text_params)

    # hide frames, set xlimit & ylimit, set title
    limit = 4
    axis.grid(False)
    axis.spines['left'].set_visible(False)
    axis.spines['right'].set_visible(False)
    axis.spines['bottom'].set_visible(False)
    axis.spines['top'].set_visible(False)
 
    axis.set_xlim([-limit, limit])
    axis.set_ylim([-limit, limit])
    #axis.set_aspect('equal')      

    if title is not None:
        if show_values:
            axis.set_title(title + "(basis vectors:[{},{}], [{},{}])".format(round(I[0],1), round(I[1],1), round(J[0],1), round(J[1],1)))
        else:
            axis.set_title(title)
    #axis.spines['left'].set_position('center')
    #axis.spines['bottom'].set_position('center')
    #axis.spines['left'].set_linewidth(0.3)
    #axis.spines['bottom'].set_linewidth(0.3)
    #axis.spines['right'].set_color('none')
    #axis.spines['top'].set_color('none')

def plot_2d_linear_transformation(vectors, matrix, show_values=False):
    figure, (axis1, axis2) = plt.subplots(1, 2)
    plot_2d_transformation_helper(axis1, np.identity(2), vectors, title='Before transformation', show_values=show_values)
    plot_2d_transformation_helper(axis2, matrix, vectors, title='After transformation', show_values=show_values)

def plot_2d_linear_transformations(vectors, *matrices, show_values=False):
    nplots = len(matrices) + 1
    nx = 2
    ny = math.ceil(nplots/nx)
    figure, axes = plt.subplots(nx, ny)
    plt.subplots_adjust(wspace=0.1, hspace=0.3)

    for i in range(nplots):  # fig_idx 
        if i == 0:
            matrix_trans = np.identity(2)
            title = 'Before transformation'
        else:
            matrix_trans = matrices[i-1] @ matrix_trans
            if i == 1:
                title = 'After {} transformation'.format(i)
            else:
                title = 'After {} transformations'.format(i)
        plot_2d_transformation_helper(axes[i//nx, i%nx], matrix_trans, vectors, title=title, show_values=show_values)
    # hide axes of the extra subplot (only when nplots is an odd number)
    if nx*ny > nplots:
        axes[-1,-1].axis('off')

def plot_2d_eigen_decomposition_transformation(v, A, show_values=False):
    e_values, e_vectors = np.linalg.eig(A)
    C = e_vectors
    print(C)
    D = np.diag(e_values)
    print(D)
    C_inv = np.linalg.inv(e_vectors)
    print(C_inv)
    A_decomp = C @ D @ C_inv
    print(A_decomp)
    plot_2d_linear_transformations(v, C_inv, D, C, show_values=show_values)

def plot_2d_svd_transformation(v, A, show_values=False):
    U, S, VT = np.linalg.svd(A)
    S = np.diag(S)
    print(U)
    print(S)
    print(VT)
    A_decomp = U @ S @ VT
    print(A_decomp)
    plot_2d_linear_transformations(v, VT, S, U, show_values=show_values)

def plot_3d_transformation_helper(axis, matrix, vectors, title=None):
    assert matrix.shape == (3,3), "the input matrix must have a shape of (3,3)"
    xcolor, ycolor, zcolor = '#0084b6', '#d8a322', '#FF3333'
    linewidth = 0.7

    grid_range = 2
    x = np.arange(-grid_range, grid_range+1)
    X, Y, Z = np.meshgrid(x,x,x)
    X_new = matrix[0,0]*X + matrix[0,1]*Y + matrix[0,2]*Z
    Y_new = matrix[1,0]*X + matrix[1,1]*Y + matrix[1,2]*Z
    Z_new = matrix[2,0]*X + matrix[2,1]*Y + matrix[2,2]*Z
    
    # draw grid lines
    for i in range(x.size):
       for j in range(x.size):
            axis.plot(X_new[:,i,j], Y_new[:,i,j], Z_new[:,i,j], c=xcolor, **grid_params)
            axis.plot(X_new[i,:,j], Y_new[i,:,j], Z_new[i,:,j], c=ycolor, **grid_params)
            axis.plot(X_new[i,j,:], Y_new[i,j,:], Z_new[i,j,:], c=zcolor, **grid_params)
    
    origin = np.zeros(1)
    # draw vectors
    color_cycle = cycle([pink, darkblue, orange, purple, brown])
    for vector in vectors:
            color = next(color_cycle)
            vector_ = matrix @ vector.reshape(-1,1)
            axis.quiver(origin, origin, origin, [vector_[0]], [vector_[1]], [vector_[2]], color=color)

    limit = 0
    for array in (X_new, Y_new, Z_new):
            limit_ = np.max(np.abs(array))
            limit = max(limit, limit_)

    axis.grid(False)
    axis.xaxis.line.set_lw(0.)
    axis.yaxis.line.set_lw(0.)
    axis.zaxis.line.set_lw(0.)
    axis.xaxis.pane.fill = False
    axis.yaxis.pane.fill = False
    axis.zaxis.pane.fill = False
    axis.xaxis.pane.set_edgecolor('white')
    axis.yaxis.pane.set_edgecolor('white')
    axis.zaxis.pane.set_edgecolor('white')

    axis.set_xlim(-limit, limit)
    axis.set_ylim(-limit, limit)
    axis.set_zlim(-limit, limit)
    #axis.set_aspect('equal')      

    if title is not None:
        axis.set_title(title)  
    # adjust the whitespace between ticks and axes to get a tighter plot
    for axis_str in ['x', 'y', 'z']:
        axis.tick_params(axis=axis_str, pad=-3)    

def plot_3d_linear_transformation(vectors, matrix):
    figure = plt.figure()
    axis1 = figure.add_subplot(1, 2, 1, projection='3d')
    axis2 = figure.add_subplot(1, 2, 2, projection='3d')
    plot_3d_transformation_helper(axis1, np.identity(3), vectors, title='before transformation')
    plot_3d_transformation_helper(axis2, matrix, vectors, title='after transformation')

def plot_3d_linear_transformations(vectors, *matrices):
    nplots = len(matrices) + 1
    nx = 2                 # number of figures per row
    ny = math.ceil(nplots/nx)   # number of figures per column
    figure = plt.figure()

    for i in range(nplots):  # fig_idx
        axis = figure.add_subplot(ny, nx, i+1, projection='3d')
        if i == 0:
            matrix_trans = np.identity(3)
            title = 'Before transformation'
        else:
            matrix_trans = matrices[i-1] @ matrix_trans
            if i == 1:
                title = 'After {} transformation'.format(i)
            else:
                title = 'After {} transformations'.format(i)
        plot_3d_transformation_helper(axis, matrix_trans, vectors, title=title)

def plot_3d_eigen_decomposition_transformation(v, A):
    e_values, e_vectors = np.linalg.eig(A)
    C = e_vectors
    print(C)
    D = np.diag(e_values)
    print(D)
    C_inv = np.linalg.inv(e_vectors)
    print(C_inv)
    A_decomp = C @ D @ C_inv
    print(A_decomp)
    plot_3d_linear_transformations(v, C_inv, D, C)

def plot_3d_svd_transformation(v, A):
    U, S, VT = np.linalg.svd(A)
    S = np.diag(S)
    print(U)
    print(S)
    print(VT)
    A_decomp = U @ S @ VT
    print(A_decomp)
    plot_3d_linear_transformations(v, VT, S, U)

def plot_2d_basis_change_helper(axis, matrix, vectors, show_values, title=None):
    assert matrix.shape == (2,2), "the input matrix must have a shape of (2,2)"
    grid_range = 20
    x = np.arange(-grid_range, grid_range+1)
    X_, Y_ = np.meshgrid(x,x)
    I = matrix[:,0]
    J = matrix[:,1]
    X = I[0]*X_ + J[0]*Y_
    Y = I[1]*X_ + J[1]*Y_
    origin = np.zeros(1)
    
    # draw grid lines
    for i in range(x.size):
        axis.plot(X[i,:], Y[i,:], c=gold, **grid_params)
        axis.plot(X[:,i], Y[:,i], c=lightblue, **grid_params)
    
    #draw unit vectors
    axis.quiver(origin, origin, [I[0]], [I[1]], color=green, **quiver_params)
    axis.quiver(origin, origin, [J[0]], [J[1]], color=red, **quiver_params)
    if show_values:
        axis.text(I[0]+0.1, I[1]+0.1, '$({},{})$'.format(round(I[0],1), round(I[1],1) ), color=gold, **text_params)
        axis.text(J[0]+0.1, J[1]+0.1, '$({},{})$'.format(round(J[0],1), round(J[1],1) ), color=gold, **text_params)

    # draw vectors
    color_cycle = cycle([pink, darkblue, orange, purple, brown])
    minv = np.linalg.inv(matrix)
    for vector in vectors:
            color = next(color_cycle)
            vector_ = minv @ vector.reshape(-1,1)
            axis.quiver(origin, origin, [vector_[0]], [vector_[1]], color=color, **quiver_params)
            if show_values:
                axis.text(vector_[0]+0.1, vector_[1]+0.1, '$({},{})$'.format(np.round(vector_[0][0],1), np.round(vector_[1][0],1) ), color=gold, **text_params)

    # hide frames, set xlimit & ylimit, set title
    limit = 4
    axis.grid(False)
    axis.spines['left'].set_visible(False)
    axis.spines['right'].set_visible(False)
    axis.spines['bottom'].set_visible(False)
    axis.spines['top'].set_visible(False)
 
    axis.set_xlim([-limit, limit])
    axis.set_ylim([-limit, limit])
    #axis.set_aspect('equal')      

    if title is not None:
        if show_values:
            axis.set_title(title + "(basis vectors:[{},{}], [{},{}])".format(round(I[0],1), round(I[1],1), round(J[0],1), round(J[1],1)))
        else:
            axis.set_title(title)

def plot_2d_basis_change(vectors, matrix, show_values=False):
    figure, (axis1, axis2) = plt.subplots(1, 2)
    plot_2d_basis_change_helper(axis1, np.identity(2), vectors, title='Standard basis', show_values=show_values)
    plot_2d_basis_change_helper(axis2, matrix, vectors, title='After basis change', show_values=show_values)
    