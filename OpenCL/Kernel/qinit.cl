__kernel void square(__global float* d_p, 
                     __global float* d_u,
                     __global float* d_o,
                     const int count)
{
    int i = get_local_id(0);
//    d_o[i] = i;    
    if (i < count - 2 && i > 1)
    { 
        float p = d_p[i], u = d_u[i], 
               pl = d_p[i-1], ul = d_u[i-1], 
               pr = d_p[i+1], ur = d_u[i+1];
        barrier(CLK_GLOBAL_MEM_FENCE);        
       /* problem data */
        float rho = 1.0, K = 4.0;
        float cc = sqrt(K/rho), zz = cc*rho;
        float dt = 0.1, dx = 0.1;
        
        /* left riemann problem */
        float delta[2], a1, a2, wave1[2], wave2[2], s1, s2, amdq[2];
        delta[0] = pl - p;
        delta[1] = ul - u;
        
        a1 = (-delta[0] + zz*delta[1]) / (2.0*zz);
        a2 = (delta[0] + zz*delta[1]) / (2.0*zz);

        wave1[0] = -a1*zz;
        wave1[1] = a1;
        s1 = -cc;
        amdq[0] = s1*wave1[0];
        amdq[1] = s1*wave1[1];
        
        p -= dt/dx*amdq[0];
        u -= dt/dx*amdq[1];
        
        /* right riemann problem */
        float apdq[2];

        delta[0] = p - pr;
        delta[1] = u - ur;
        a1 = (-delta[0] + zz*delta[1]) / (2.0*zz);
        a2 = (delta[0] + zz*delta[1]) / (2.0*zz);
        wave2[0] = -a2*zz;
        wave2[1] = a2;
        s2 = cc;
        apdq[0] = s2*wave2[0];
        apdq[1] = s2*wave2[1];
        
        p -= dt/dx*apdq[0];
        u -= dt/dx*apdq[1];
        
        d_p[i] = p;
        d_u[i] = u;
        d_o[i] = d_p[i];
    }
    else{
        d_o[i] = -1;
    }
}
