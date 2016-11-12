#include <iostream>
#include <cmath>

#include "OpenCL/opencl.h"
#include "helper.h"

#define DATA_SIZE (8)

int main(int argc, char const *argv[])
{
	cl_int            err;
	cl_platform_id    platform = GetPlatform(0); 
	cl_device_id      device = GetDevice(platform, 0);
	cl_context        context;
	cl_command_queue  commands;
	cl_program        program;
	cl_kernel         k_acoustic_1d;

	/* Create context */
    context = clCreateContext(0, 1, &device, NULL, NULL, &err);
	CheckError(err);

	/* Create commands */
    commands = clCreateCommandQueue (context, device, 0, &err);
    CheckError(err);

	/* Create program from source */
	program = CreateProgram(LoadKernel ("Kernel/acoustic_1d.cl"), context);

	err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
	if (err != CL_SUCCESS)
    {
    	ProgramErrMsg(program, device);
    }

	/* Create Kernel */ 
	k_acoustic_1d = clCreateKernel(program, "acoustic_1d", &err);
	CheckError(err);

	/* Create data and run the kernel */
    int meqn = 2;
	int mx = DATA_SIZE, mbc = 2;
	int mtot = mx + 2*mbc;
	float p[mtot], u[mtot];
	float results[mtot];

    std::size_t global=mtot;
    std::size_t local =mtot;

	/* Initialize data */
	for(int i = 0; i < mtot; i++){
        if(i == 5){
        	p[i] = 2;
        	u[i] = 1;
		}
		else{
			p[i] = 0;
			u[i] = 0;
		}
	}

	/* Allocate device memory */
	cl_mem d_p, d_u;                 // device memory u

	d_p  = clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(float)*mtot, NULL, NULL);
	d_u  = clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(float)*mtot, NULL, NULL);

    if (!d_p || !d_u)
    {
        printf("Error: Failed to allocate device memory!\n");
        exit(1);
    }

	/* Write data set into the input array in device memory */
    CheckError(clEnqueueWriteBuffer(commands, d_p, CL_TRUE, 0, sizeof(float)*mtot, p, 0, NULL, NULL));
    CheckError(clEnqueueWriteBuffer(commands, d_u, CL_TRUE, 0, sizeof(float)*mtot, u, 0, NULL, NULL));

	/* Set kernel arguments */
    CheckError(clSetKernelArg(k_acoustic_1d, 0, sizeof(cl_mem), &d_p));
    CheckError(clSetKernelArg(k_acoustic_1d, 1, sizeof(cl_mem), &d_u));
    CheckError(clSetKernelArg(k_acoustic_1d, 2, sizeof(int)   , &mx));
    CheckError(clSetKernelArg(k_acoustic_1d, 3, sizeof(int)   , &mbc));

	/* Retrieve kernel work group info */
    // CheckError(clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL), 
    // 	       "Retrieve work group info");

	/* Launch kernel */
	for (int j = 0; j < 5; ++j)
	{
		std::cout<<std::endl<<"Time Step = "<<j<<std::endl;
		CheckError(clEnqueueNDRangeKernel(commands, k_acoustic_1d, 1, NULL, &global, &local, 0, NULL, NULL));

		/* Read ouput array */
	    CheckError(clEnqueueReadBuffer(commands, d_p, CL_TRUE, 0, sizeof(float)*mtot, p, 0, NULL, NULL ));  
	    CheckError(clEnqueueReadBuffer(commands, d_u, CL_TRUE, 0, sizeof(float)*mtot, u, 0, NULL, NULL ));

		/* Print out result */
	    for (int i = 0; i < global; ++i)
	    {
	    	std::cout<<"p["<<i<<"]"<<" = "<<p[i]<<", u["<<i<<"]="<<u[i]<<std::endl;
	    }

	}

    
    clReleaseMemObject(d_u);
    clReleaseMemObject(d_p);
    clReleaseCommandQueue (commands);
    clReleaseKernel(k_acoustic_1d);
    clReleaseProgram(program);
    clReleaseContext(context);

	return 0;
}