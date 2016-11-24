__kernel void rp1_acoustic(__global double* d_q, 
                           __global double* d_apdq, 
                           __global double* d_amdq,
                           __global double* d_s,
                           const int mx, const int mbc, 
                           const double rho, const double K)
{
    /* Input: q, Output: d_apdq, d_amdq*/
    int i = get_global_id(0); 
    d_s[i] = 0;
    if (i < mx + mbc + 1 && i > mbc - 1)
    {
        /* Solve riemann problem from "ith" cell boundary 
          (the boundary of ith and (i-1)th cell) */
        
        double pl = d_q[2*(i-1)], ul = d_q[2*(i-1)+1], 
              pr = d_q[2*i], ur = d_q[2*i+1];

        double cc = sqrt(K/rho), zz = cc*rho;
        double delta[2], 
              a1, s1, wave1[2], 
              a2, s2, wave2[2]; 

        delta[0] = pr - pl;
        delta[1] = ur - ul;
        
        a1 = (-delta[0] + zz*delta[1]) / (2.0*zz);
        wave1[0] = -a1*zz;
        wave1[1] = a1;
        s1 = -cc;
        d_amdq[2*i]   = s1*wave1[0];
        d_amdq[2*i+1] = s1*wave1[1];
        
        a2 = (delta[0] + zz*delta[1]) / (2.0*zz);
        wave2[0] = a2*zz;
        wave2[1] = a2;
        s2 = cc;

        d_apdq[2*i]   = s2*wave2[0];
        d_apdq[2*i+1] = s2*wave2[1];

        d_s[i] = max(-s1, s2);
    }
}
