__kernel void qinit(__global float* d_p, 
                    __global float* d_u,
                    const int mx, const int mbc){
	int i = get_local_id(0);

	d_p[i] = 0;
	d_u[i] = 0;
	if( i == mbc ){
		d_p[i] = 0.4;
		d_u[i] = 0.2;
	}
}