__kernel void rp1_acoustic(__global float* d_p, 
                           __global float* d_u,
                           __global float* apdq, 
                           __global float* amdq,
                           const int meqn, const int mx, const int mbc, 
                           const float dt_temp, const float dx, 
                           const float rho, const float K)
{
    int i = get_local_id(0); 
    if (i < mx + mbc && i > mbc - 1)
    { 
        float pl = d_p[i-1], ul = d_u[i-1], 
              pr = d_p[i], ur = d_u[i];
       
        /* problem data */
        float cc = sqrt(K/rho), zz = cc*rho;
        float dt = dx/cc;
        
        /* left riemann problem */
        float delta[2], a1, a2, wave1[2], wave2[2], s1, s2, amdq[2];

        delta[0] = p - pl;
        delta[1] = u - ul;
        
        a2 = (delta[0] + zz*delta[1]) / (2.0*zz);

        wave1[0] = a2*zz;
        wave1[1] = a2;
        s1 = cc;
        amdq[0] = s1*wave1[0];
        amdq[1] = s1*wave1[1];
        
        p -= dt/dx*amdq[0];
        u -= dt/dx*amdq[1];
        
        /* right riemann problem */
        float apdq[2];

        delta[0] = pr - p;
        delta[1] = ur - u;
        a1 = (-delta[0] + zz*delta[1]) / (2.0*zz);
        wave2[0] = -a1*zz;
        wave2[1] = a1;
        s2 = -cc;

        apdq[0] = s2*wave2[0];
        apdq[1] = s2*wave2[1];
        
        p -= dt/dx*apdq[0];
        u -= dt/dx*apdq[1];
        
        d_p[i] = p;
        d_u[i] = u;
    }
}
