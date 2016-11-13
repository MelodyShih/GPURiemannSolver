#include <iostream>
#include <cmath>
#include <cassert>
#include <fstream>

#include "OpenCL/opencl.h"
#include "helper.h"

int main(int argc, char const *argv[])
{
	/* Opencl related variables */
	cl_int            err;
	cl_platform_id    platform = GetPlatform(0); 
	cl_device_id      device = GetDevice(platform, 0);
	cl_context        context;
	cl_command_queue  commands;
	cl_program        p_acoustic_1d, p_qinit;
	cl_kernel         k_acoustic_1d;
	cl_kernel         k_qinit;
	cl_mem            d_p, d_u;

	/* Data for pde solver */
    int meqn = 2;
    int ndim = 1;
    int maux = 0;
	int mx = 10, mbc = 2;
	int mtot = mx + 2*mbc;
	int iframe = 0;
	float xlower = 0.0;
	float xupper = 1.0;
	float dx = (xupper - xlower)/mx;
	float t = 0;
	float dt = 0.05;
	float p[mtot], u[mtot];

    std::size_t global=mtot;
    std::size_t local =mtot;

	/* Create context */
    context = clCreateContext(0, 1, &device, NULL, NULL, &err);
	CheckError(err);

	/* Create commands */
    commands = clCreateCommandQueue (context, device, 0, &err);
    CheckError(err);

	/* Create program, kernel from source */
	p_acoustic_1d = CreateProgram(LoadKernel ("Kernel/acoustic_1d.cl"), context);
	err     = clBuildProgram(p_acoustic_1d, 1, &device, NULL, NULL, NULL);
	if (err != CL_SUCCESS)
    {
    	ProgramErrMsg(p_acoustic_1d, device);
    }
	k_acoustic_1d = clCreateKernel(p_acoustic_1d, "acoustic_1d", &err);


	p_qinit = CreateProgram(LoadKernel ("Kernel/qinit.cl"), context);
	err     = clBuildProgram(p_qinit, 1, &device, NULL, NULL, NULL);
	if (err != CL_SUCCESS)
    {
    	ProgramErrMsg(p_qinit, device);
    }
	k_qinit = clCreateKernel(p_qinit, "qinit", &err);
	CheckError(err);

	/* Allocate device memory */
	d_p  = clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(float)*mtot, NULL, NULL);
	d_u  = clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(float)*mtot, NULL, NULL);
	assert(d_p != 0);
	assert(d_u != 0);

	/* Write data set into the input array in device memory */
    CheckError(clEnqueueWriteBuffer(commands, d_p, CL_TRUE, 0, sizeof(float)*mtot, p, 0, NULL, NULL));
    CheckError(clEnqueueWriteBuffer(commands, d_u, CL_TRUE, 0, sizeof(float)*mtot, u, 0, NULL, NULL));

	/* QINIT */
    CheckError(clSetKernelArg(k_qinit, 0, sizeof(cl_mem), &d_p));
    CheckError(clSetKernelArg(k_qinit, 1, sizeof(cl_mem), &d_u));
    CheckError(clSetKernelArg(k_qinit, 2, sizeof(int)   , &mx));
    CheckError(clSetKernelArg(k_qinit, 3, sizeof(int)   , &mbc));

    CheckError(clEnqueueNDRangeKernel(commands, k_qinit, 1, NULL, &global, &local, 0, NULL, NULL));

	CheckError(clEnqueueReadBuffer(commands, d_p, CL_TRUE, 0, sizeof(float)*mtot, p, 0, NULL, NULL ));  
	CheckError(clEnqueueReadBuffer(commands, d_u, CL_TRUE, 0, sizeof(float)*mtot, u, 0, NULL, NULL ));

	for (int i = 0; i < mtot; ++i)
	{
	    	std::cout<<"p["<<i<<"] = "<<p[i]<<std::endl;
	}
	out1(meqn, mbc, mx, xlower, dx, p, u, 0.0, iframe, NULL, maux);
	iframe++;
	/* Periodic BC */

	/* ADVANCE SOLUTION */
    CheckError(clSetKernelArg(k_acoustic_1d, 0, sizeof(cl_mem), &d_p));
    CheckError(clSetKernelArg(k_acoustic_1d, 1, sizeof(cl_mem), &d_u));
    CheckError(clSetKernelArg(k_acoustic_1d, 2, sizeof(int)   , &mx));
    CheckError(clSetKernelArg(k_acoustic_1d, 3, sizeof(int)   , &mbc));

	/* Launch kernel */
	for (int j = 0; j < 20; ++j)
	{
		t = t + dt;
		CheckError(clEnqueueNDRangeKernel(commands, k_acoustic_1d, 1, NULL, &global, &local, 0, NULL, NULL));

		/* Read ouput array */
	    CheckError(clEnqueueReadBuffer(commands, d_p, CL_TRUE, 0, sizeof(float)*mtot, p, 0, NULL, NULL ));  
	    CheckError(clEnqueueReadBuffer(commands, d_u, CL_TRUE, 0, sizeof(float)*mtot, u, 0, NULL, NULL ));
	    std::cout<<std::endl;
	    for (int i = 0; i < mtot; ++i)
	    {
	    	std::cout<<"p["<<i<<"] = "<<p[i]<<std::endl;
	    }
	    out1(meqn, mbc, mx, xlower, dx, p, u, t, iframe, NULL, maux);
	    iframe++;
	}
    
    clReleaseMemObject(d_u);
    clReleaseMemObject(d_p);
    clReleaseCommandQueue (commands);
    clReleaseKernel(k_acoustic_1d);
    clReleaseKernel(k_qinit);
    clReleaseProgram(p_acoustic_1d);
    clReleaseProgram(p_qinit);
    clReleaseContext(context);

	return 0;
}