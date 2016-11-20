__kernel void update_q1(__global float* d_q, 
                        __global float* d_apdq, 
                        __global float* d_amdq,
                        const int meqn, const int mx, const int mbc, 
                        const float dx, const float dt)
{
	int i = get_global_id(0); 
    if (i < mx + mbc && i > mbc - 1)
    {
    	for(int m=0; m<meqn; m++){
    		d_q[meqn*i+m] -= dt/dx*(d_apdq[meqn*i+m] 
    			                  + d_amdq[meqn*(i+1)+m]);
    	} 
    }
}