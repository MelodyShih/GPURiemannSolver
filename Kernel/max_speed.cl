__kernel void max_speed(__global double* d_speed,
					    __local  double* s_speed)
{
	int i = get_global_id(0);
	int tid = get_local_id(0);
	s_speed[tid] = d_speed[i];
	barrier(CLK_LOCAL_MEM_FENCE);
		
	for(unsigned int s=(get_local_size(0) - 1)/2 + 1; s > 0; s>>=1) {
		if (tid < s) 
			s_speed[tid] = max(s_speed[tid + s], s_speed[tid]);
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	if(tid == 0) d_speed[get_group_id(0)] = s_speed[tid];
}