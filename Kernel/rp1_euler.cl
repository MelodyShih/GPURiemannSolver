__kernel void rp1_euler(__global double* d_q, 
                        __global double* d_apdq, 
                        __global double* d_amdq,
                        __global double* d_s,
                        const int mx, const int mbc, 
                        const double gamma)
{
    /* Input: q, Output: d_apdq, d_amdq, d_s */
    /* meqn = 3, mwaves = 3 */
    /* Conserved quantities: density(rho), momentum(rhou), energy(e) */
    int i = get_global_id(0); 
    d_s[i] = 0;
    if (i < mx + mbc + 1 && i > mbc - 1)
    {
        /* Solve riemann problem from "ith" cell boundary 
          (the boundary of ith and (i-1)th cell) */
        
        double rhol = d_q[3*(i-1)], rhoul = d_q[3*(i-1)+1], el = d_q[3*(i-1)+2],
               rhor = d_q[3*i], rhour = d_q[3*i+1], er = d_q[3*i+2];

        double gamma1 = gamma - 1.0;
        double delta[3], 
              a1, s1, wave1[3], 
              a2, s2, wave2[3], 
              a3, s3, wave3[3]; 
        double u, a, enth;
        double rhosqrtl, rhosqrtr, pl, pr, rhosq2;
        double temp;

        /* Calculate roe average */
        rhosqrtl = sqrt(rhol);
        rhosqrtr = sqrt(rhor);
        pl = gamma1*(el - 0.5*pow(rhoul, 2)/rhol);
        pr = gamma1*(er - 0.5*pow(rhour, 2)/rhor);
        rhosq2 = rhosqrtl + rhosqrtr;
        u = (rhoul/rhosqrtl + rhour/rhosqrtr)/rhosq2;
        enth = ((el + pl)/rhosqrtl + (er + pr)/rhosqrtr)/rhosq2;
        a = sqrt(gamma1*(enth - 0.5*pow(u, 2)));
        
        /* find coefficients of the 3 eigenvectors */
        delta[0] = rhor - rhol;
        delta[1] = rhour - rhoul;
        delta[2] = er - el;
        a2 = gamma1/pow(a, 2) * ((enth - pow(u, 2))*delta[0] + u*delta[1] - delta[2]);
        a3 = (delta[1] + (a-u)*delta[0] - a*a2)/(2.0*a);
        a1 = delta[0] - a2 - a3;

        /* Compute the waves */
        wave1[0] = a1;
        wave1[1] = a1 * (u-a);
        wave1[2] = a1 * (enth - u*a);
        s1 = u - a;

        wave2[0] = a2;
        wave2[1] = a2 * u;
        wave2[2] = a2 * 0.5 * pow(u, 2);
        s2 = u;

        wave3[0] = a3;
        wave3[1] = a3 * (u+a);
        wave3[2] = a3 * (enth + u*a);
        s3 = u + a;

        /* Compute Godunov flux */
        d_amdq[3*i]   = 0.0;
        d_amdq[3*i+1] = 0.0;
        d_amdq[3*i+2] = 0.0;
        d_apdq[3*i]   = 0.0;
        d_apdq[3*i+1] = 0.0;
        d_apdq[3*i+2] = 0.0;
        
        if (s1 < 0){
            d_amdq[3*i]   += s1*wave1[0];
            d_amdq[3*i+1] += s1*wave1[1];
            d_amdq[3*i+2] += s1*wave1[2];
        }else{
            d_apdq[3*i]   += s1*wave1[0];
            d_apdq[3*i+1] += s1*wave1[1];
            d_apdq[3*i+2] += s1*wave1[2];
        }

        if (s2 < 0){
            d_amdq[3*i]   += s2*wave2[0];
            d_amdq[3*i+1] += s2*wave2[1];
            d_amdq[3*i+2] += s2*wave2[2];
        }else{
            d_apdq[3*i]   += s2*wave2[0];
            d_apdq[3*i+1] += s2*wave2[1];
            d_apdq[3*i+2] += s2*wave2[2];
        }

        if (s3 < 0){
            d_amdq[3*i]   += s3*wave3[0];
            d_amdq[3*i+1] += s3*wave3[1];
            d_amdq[3*i+2] += s3*wave3[2];
        }else{
            d_apdq[3*i]   += s3*wave3[0];
            d_apdq[3*i+1] += s3*wave3[1];
            d_apdq[3*i+2] += s3*wave3[2];
        }
        temp = max(fabs(s1), fabs(s2));
        d_s[i] = max(temp, fabs(s3));
    }
}
