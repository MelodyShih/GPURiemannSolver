__kernel void calc_cfl(__global float* d_s,
					   __global float* d_cfl,
                       const float dx, const float dt,
                       const int mx)
{
	int i = get_global_id(0);
	if(i==0) *d_cfl = d_s[0]*dt/dx;
}