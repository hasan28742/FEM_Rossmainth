import distmesh2d as dm
import distance_functions as distfunc
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import boundary as bnd;
import scipy.sparse
import scipy.sparse.linalg
import assemble;
matplotlib.rcParams.update({'font.size': 14, 'font.family': 'sans-serif'});

# rectangle with circular hole, refined at circle boundary:
fd = lambda p: distfunc.ddiff(distfunc.drectangle(p,-1,1,-1,1),distfunc.dcircle(p,0,0,0.5));
p,t = dm.distmesh2d(fd, distfunc.huniform, 0.04, (-1,-1,1,1),[(-1,-1),(-1,1),(1,-1),(1,1)]);
p,t,num_int_nodes,num_bnd_nodes,area = bnd.bnd2d(fd,p,t,-1.0,-1.0,1.0,1.0);
num_elems = t.shape[0];
num_nodes = p.shape[0];
total_area = area.sum();

# poisson assemble and solve
def frhs(x,y):
    return 1.0;
A = assemble.stiffness_matrix(p,t,num_int_nodes,area);
b = assemble.rhs(p,t,num_int_nodes,area,frhs);
U = np.zeros(num_int_nodes+num_bnd_nodes, dtype=np.float64);
U[0:num_int_nodes] = scipy.sparse.linalg.spsolve(A,b);

xticks = np.linspace(-1,1,5);
yticks = np.linspace(-1,1,5);

# plot mesh
zfaces = np.ones(num_elems);
plt.figure(1)
plt.clf();
plt.gca().set_aspect('equal')
plt.gca().set_xlim([-1.05,1.05])
plt.gca().set_ylim([-1.05,1.05])
plt.xticks(xticks)
plt.yticks(yticks)
plt.tripcolor(p[:,0],
              p[:,1],
              t,
              facecolors=zfaces,
              edgecolors='k',
              cmap='inferno',
              vmin=0.00,vmax=1.00);
plt.title('PyDistMesh (int_nodes = %i, bnd_nodes = %i)' % (num_int_nodes,num_bnd_nodes));
plt.xlabel('x')
plt.ylabel('y')

# plot solution
plt.figure(2)
plt.clf();
plt.gca().set_aspect('equal')
plt.gca().set_xlim([-1.05,1.05])
plt.gca().set_ylim([-1.05,1.05])
plt.xticks(xticks)
plt.yticks(yticks)
plt.tripcolor(p[:,0],
              p[:,1],
              t,
              U,
              shading='flat',
              cmap='inferno');
plt.title('cG(1) Poisson solution: pcolor plot');
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar()

# plot solution
plt.figure(3)
plt.clf();
plt.gca().set_aspect('equal')
plt.gca().set_xlim([-1.05,1.05])
plt.gca().set_ylim([-1.05,1.05])
plt.xticks(xticks)
plt.yticks(yticks)
plt.tricontour(p[:,0],
               p[:,1],
               t,
               U,
               colors='k',
               levels=10);
plt.tripcolor(p[:,0],
              p[:,1],
              t,
              U,
              shading='flat',
              cmap='inferno');
plt.title('cG(1) Poisson solution: pcolor + contours');
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar()


# plot solution

plt.show()
