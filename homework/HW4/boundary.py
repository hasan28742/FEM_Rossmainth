def bnd2d(fd,p,t,xmin,ymin,xmax,ymax):

    import numpy as np;
    from scipy.spatial import Delaunay;
    geps = 1.0e-8;

    # get unique nodes
    r2 = (p[:,0]-xmin+geps)/(xmax-xmin) + 1.0e6*(p[:,1]-ymin+geps)/(ymax-ymin);
    r2,indeces = np.unique(r2,return_index=True);
    p = p[indeces,:];

    # retriangulate the new ordering of points
    t = Delaunay(p).simplices;
    pmid = p[t].sum(1)/3;              # Compute centroids
    t = t[fd(pmid) < -geps];           # Keep interior triangles

    # find all boundary nodes and put them at the end of the node list
    num_nodes = p.shape[0];
    ftmp = fd(p);
    pnew = np.zeros((num_nodes,2));
    k = 0;
    kint = 0;
    kbnd = num_nodes-1;

    for i in range(0,num_nodes):
        tmp = p[i,:];
        if np.fabs(ftmp[i])<=1.0e-8:
            k=k+1;
            pnew[kbnd,:] = tmp;
            kbnd = kbnd-1;
        else:
            pnew[kint,:] = tmp;
            kint = kint+1;
    num_bnd_nodes = k;
    num_int_nodes = num_nodes - num_bnd_nodes;

    # retriangulate the new ordering of points
    tnew = Delaunay(pnew).simplices;
    pmid = pnew[tnew].sum(1)/3;        # Compute centroids
    tnew = tnew[fd(pmid) < -geps];     # Keep interior triangles

    # compute areas
    num_elems = tnew.shape[0];
    area = np.zeros(num_elems);
    for i in range(0,num_elems):
        tt = tnew[i,:];
        x1 = pnew[tt[0],0];
        y1 = pnew[tt[0],1];
        x2 = pnew[tt[1],0];
        y2 = pnew[tt[1],1];
        x3 = pnew[tt[2],0];
        y3 = pnew[tt[2],1];
        atmp = 0.5*(x1*(y2-y3)+x2*(y3-y1)+x3*(y1-y2));
        if atmp<0.0:
            tnew[i,1] = tt[2];
            tnew[i,2] = tt[1];
        area[i] = np.fabs(atmp);

    return pnew,tnew,num_int_nodes,num_bnd_nodes,area;
