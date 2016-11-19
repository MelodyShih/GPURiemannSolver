__kernel void qinit(__global float* d_q, 
                    const int meqn, const int mx, const int mbc, 
                    const float xlower, const float dx){
    int i = get_local_id(0);
    float beta = 200;
    float xcell;
    
    /* Right going wave */
    for(int m = 0; m<meqn; m++){
        d_q[meqn*i] = 0;
        d_q[meqn*i+m] = 0;
        if( i == 2 ){
            d_q[meqn*i] = 0.4;
            d_q[meqn*i+m] = 0.2;
        }
    }

    /* gaussian hump */ 
    /*
    i = i - mbc + 1;
    xcell = xlower + (i-0.5)*dx;
    d_p[i] = exp(-beta * pow((xcell-0.0),2));
    d_u[i] = 0.0;*/
}