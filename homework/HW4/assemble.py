def stiffness_matrix(p,t,NIN,area):

    import numpy as np;
    from scipy.sparse import lil_matrix

    A = lil_matrix((NIN,NIN), dtype=np.float64);
    num_elems = t.shape[0];

    for i in range(0,num_elems):
        tt = t[i,:];
        x1 = p[tt[0],:];
        x2 = p[tt[1],:];
        x3 = p[tt[2],:];

        M = np.array([[1.0, x1[0], x1[1]],
                      [1.0, x2[0], x2[1]],
                      [1.0, x3[0], x3[1]]]);
        Minv = np.linalg.inv(M);

        # hat functions
        phi1 = np.dot(Minv,np.array([1.0,0.0,0.0]));
        phi2 = np.dot(Minv,np.array([0.0,1.0,0.0]));
        phi3 = np.dot(Minv,np.array([0.0,0.0,1.0]));

        # gradient of hat functions
        gphi = np.array([[phi1[1],phi1[2]],
                         [phi2[1],phi2[2]],
                         [phi3[1],phi3[2]]]);

        # stiffness matrix
        for j in range(0,3):
            for k in range(0,3):
                if (tt[j]<NIN and tt[k]<NIN):
                    A[tt[j],tt[k]] += area[i]*np.dot(gphi[j,:],gphi[k,:]);

    A = A.tocsr();
    return A;


def rhs(p,t,NIN,area,f):

    import numpy as np;

    b = np.zeros(NIN,dtype=np.float64);

    num_elems = t.shape[0];
    for i in range(0,num_elems):
        tt = t[i,:];
        x1 = p[tt[0],:];
        x2 = p[tt[1],:];
        x3 = p[tt[2],:];

        # rhs
        if (tt[0]<NIN):
            b[tt[0]] += area[i]/3.0*f(x1[0],x1[1]);

        if (tt[1]<NIN):
            b[tt[1]] += area[i]/3.0*f(x2[0],x2[1]);

        if (tt[2]<NIN):
            b[tt[2]] += area[i]/3.0*f(x3[0],x3[1]);

    return b;
