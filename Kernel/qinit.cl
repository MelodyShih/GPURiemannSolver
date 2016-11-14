__kernel void qinit(__global float* d_p, 
                    __global float* d_u,
                    const int mx, const int mbc){
    int i = get_local_id(0);

    d_p[i] = 0;
    d_u[i] = 0;
    if( i == 8 ){
        d_p[i] = 0.4;
        d_u[i] = 0.4;
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
    if( i == 0 ){
        d_p[i] = d_p[mx + mbc];
        d_u[i] = d_u[mx + mbc];
    }
    if( i == 1 ){
        d_p[i] = d_p[mx + mbc + 1];
        d_u[i] = d_u[mx + mbc + 1];
    }
    if( i == mx + 2*mbc-1 ){
        d_p[i] = d_p[mbc + 1];
        d_u[i] = d_u[mbc + 1];
    }
    if( i == mx + 2*mbc-2 ){
        d_p[i] = d_p[mbc];
        d_u[i] = d_u[mbc];
    }
}