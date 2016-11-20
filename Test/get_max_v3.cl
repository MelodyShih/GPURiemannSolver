__kernel void get_max(__global float* d_data,
					  __local  float* s_data, 
					  const int arraylength)
{
	/*
		Reference: http://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf
	*/
	int i = get_global_id(0);
	if( i < arraylength ){
		int tid = get_local_id(0);
		s_data[tid] = d_data[i];
		barrier(CLK_LOCAL_MEM_FENCE);
		
		for(unsigned int s=(get_local_size(0) - 1)/2 + 1; s > 0; s>>=1) {
			if (tid < s) 
				s_data[tid] = max(s_data[tid + s], s_data[tid]);
			barrier(CLK_LOCAL_MEM_FENCE);
		}
		if(tid == 0) d_data[get_group_id(0)] = s_data[tid];
	}
}