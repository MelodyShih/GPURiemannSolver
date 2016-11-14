__kernel void bc1(__global float* d_p, 
                  __global float* d_u,
                  const int mx, const int mbc){
    int i = get_local_id(0);
    barrier(CLK_GLOBAL_MEM_FENCE);
    /* Periodic BC */
    /* Left */
    if( i < mbc){
        d_p[i] = d_p[mx + i];
        d_u[i] = d_u[mx + i];
    }
    /* Right */
    if( i > mx + mbc - 1 && i < mx + 2*mbc){
        d_p[i] = d_p[mbc + i - mx - mbc];
        d_u[i] = d_u[mbc + i - mx - mbc];
    }
}