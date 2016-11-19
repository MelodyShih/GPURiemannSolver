__kernel void bc1(__global float* d_q, 
                  const int meqn, const int mx, const int mbc){
    int i = get_local_id(0);
    /* Periodic BC */
    /* Left */
    if( i < mbc){
        for(int m = 0; m<meqn; m++){
            d_q[meqn*i + m] = d_q[meqn*(mx + i)+m];
        }
    }
    /* Right */
    if( i > mx + mbc - 1 && i < mx + 2*mbc){
        for(int m = 0; m<meqn; m++){
            d_q[meqn*i + m] = d_q[meqn*(i - mx)+m];
        }
    }
}