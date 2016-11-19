__kernel void acoustic_1d(__global float* d_q, 
                          const int mx, const int mbc, 
                          const float dt_temp, const float dx, 
                          const float rho, const float K)
{
    /* meqn = 2, mwaves = 2, 
       Conserved quantities: pressure(p), velocity(u) */

    int i = get_local_id(0); 
    barrier(CLK_GLOBAL_MEM_FENCE); 
    if (i < mx + mbc && i > mbc - 1)
    {
        float p  = d_q[2*i], u = d_q[2*i+1], 
              pl = d_q[2*(i-1)], ul = d_q[2*(i-1)+1], 
              pr = d_q[2*(i+1)], ur = d_q[2*(i+1)+1];

        float cc = sqrt(K/rho), zz = cc*rho;
        float dt = dx/cc; // Should be removed and use the input dt
        float delta[2], 
              a1, s1, wave1[2], // right-going wave 
              a2, s2, wave2[2]; //  left-going wave

        /* left riemann problem */
        delta[0] = p - pl;
        delta[1] = u - ul;
        a2 = (delta[0] + zz*delta[1]) / (2.0*zz);
        wave1[0] = a2*zz;
        wave1[1] = a2;
        s1 = cc;
        
        p -= dt/dx*s1*wave1[0];
        u -= dt/dx*s1*wave1[1];
        
        /* right riemann problem */
        delta[0] = pr - p;
        delta[1] = ur - u;
        a1 = (-delta[0] + zz*delta[1]) / (2.0*zz);
        wave2[0] = -a1*zz;
        wave2[1] = a1;
        s2 = -cc;

        p -= dt/dx*s2*wave2[0];
        u -= dt/dx*s2*wave2[1];
        
        d_q[2*i] = p;
        d_q[2*i + 1] = u;
    }
}
