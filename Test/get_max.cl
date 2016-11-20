__kernel void get_max(__global float* d_data,
					  __global float* d_maxvalue,
					  __local float* s_data)
{
	int i = get_global_id(0);
	int tid = get_local_id(0);
	s_data[tid] = d_data[i];
	barrier(CLK_LOCAL_MEM_FENCE);
	
	for(unsigned int s=1; s < get_local_size(0); s *= 2) {
		if (tid % (2*s) == 0) 
			s_data[tid] = max(s_data[tid + s], s_data[tid]);
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if(i==0) *d_maxvalue = s_data[i];
}