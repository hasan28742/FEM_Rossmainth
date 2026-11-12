import distmesh2d as dm
import distance_functions as distfunc
import numpy as np
import matplotlib.pyplot as plt
import boundary as bnd;

# Rectangle with circular hole, refined at circle boundary:
fd = lambda p: distfunc.ddiff(distfunc.drectangle(p,-1,1,-1,1),distfunc.dcircle(p,0,0,0.5));
fh = lambda p: 0.1+0.3*distfunc.dcircle(p,0,0,0.5);
p,t = dm.distmesh2d(fd, fh, 0.04, (-1,-1,1,1),[(-1,-1),(-1,1),(1,-1),(1,1)]);
p,t,num_int_nodes,num_bnd_nodes,area = bnd.bnd2d(fd,p,t,-1.0,-1.0,1.0,1.0);

num_elems = t.shape[0];
num_nodes = p.shape[0];
total_area = area.sum();

# plot
zfaces = np.zeros(num_elems);
plt.figure(1)
plt.clf();
plt.gca().set_aspect('equal')
plt.gca().set_xlim([-1.1,1.1])
plt.gca().set_ylim([-1.1,1.1])
plt.tripcolor(p[:,0],
              p[:,1],
              t,
              facecolors=zfaces,
              edgecolors='k',
              cmap='RdPu');

# interior nodes
plt.plot(p[0:num_int_nodes,0],p[0:num_int_nodes,1],'b.');

# boundary nodes
plt.plot(p[num_int_nodes:,0],p[num_int_nodes:,1],'r.');

plt.title('PyDistMesh (elems = %i, int_nodes = %i, bnd_nodes = %i)' % (num_elems,num_int_nodes,num_bnd_nodes));
plt.xlabel('x')
plt.ylabel('y')

# plot
plt.figure(2)
plt.clf();
plt.gca().set_aspect('equal')
plt.gca().set_xlim([-1.1,1.1])
plt.gca().set_ylim([-1.1,1.1])
plt.tripcolor(p[:,0],
              p[:,1],
              t,
              facecolors=area,
              edgecolors='k',
              cmap='RdPu');
plt.title('PyDistMesh (total_area = %0.4e)' % (total_area));
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar();

plt.show()
