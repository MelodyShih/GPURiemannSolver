__kernel void qinit(__global float* d_p, 
                    __global float* d_u,
                    const int mx, const int mbc, 
                    const float xlower, const float dx){
    int i = get_local_id(0);
    float beta = 200;
    float xcell;
    
    /* Right going wave */
    /*
    d_p[i] = 0;
    d_u[i] = 0;
    if( i == 3 ){
        d_p[i] = 0.4;
        d_u[i] = 0.2;
    }*/

    /* gaussian hump */ 
    i = i - mbc + 1;
    xcell = xlower + (i-0.5)*dx;
    d_p[i] = exp(-beta * pow((xcell-0.3),2));
    d_u[i] = 0.0;
}