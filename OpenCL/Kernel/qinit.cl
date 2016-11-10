void RP(__global double *p, __global double *u, 
        const double pr, const double ur,
        const double pl, const double ul)
{
    /* problem data */
    double rho = 1.0, K = 4.0;
    double cc = sqrt(K/rho), zz = cc*rho;
    double dt = 0.1, dx = 0.1;
    
    /* left riemann problem */
    double delta[2], a1, a2, wave1[2], wave2[2], s1, s2, amdq[2];
    delta[0] = pl - *p;
    delta[1] = ul - *u;
    
    a1 = (-delta[0] + zz*delta[1]) / (2.0*zz);
    a2 = (delta[0] + zz*delta[1]) / (2.0*zz);

    wave1[0] = -a1*zz;
    wave1[1] = a1;
    s1 = -cc;
    amdq[0] = s1*wave1[0];
    amdq[1] = s1*wave1[1];
    
    (*p) -= dt/dx*amdq[0];
    (*u) -= dt/dx*amdq[1];
    
    /* right riemann problem */
    double apdq[2];

    delta[0] = *p - pr;
    delta[1] = *u - ur;
    a1 = (-delta[0] + zz*delta[1]) / (2.0*zz);
    a2 = (delta[0] + zz*delta[1]) / (2.0*zz);
    wave2[0] = -a2*zz;
    wave2[1] = a2;
    s2 = cc;
    apdq[0] = s2*wave2[0];
    apdq[1] = s2*wave2[1];
    
    (*p) -= dt/dx*apdq[0];
    (*u) -= dt/dx*apdq[1];
}

void test(__global double *p){
    (*p)++;
}
__kernel void square(__global double* d_p, 
                     __global double* d_u,
                     __global double* d_o,
                     const int count)
{
    int i = get_global_id(0);
    if(i == 1){
        for (int timestep = 0; timestep < 10; ++timestep)
        {
            RP(&d_p[i], &d_u[i], d_p[i+1], d_u[i+1], d_p[i-1], d_u[i-1]);
            //test(&d_p[i]);
            //d_o[i] = timestep;
            // d_o[i] = RP(d_p[i], d_u[i], 
            //             d_p[i+1], d_u[i+1], 
            //             d_p[i-1], d_u[i-1]);
        }
    }
    d_o[i] = d_p[i];
}
