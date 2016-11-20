#include <iostream>
#include <iomanip>
#include <cmath>
#include <cassert>
#include <fstream>

#include "OpenCL/opencl.h"
#include "../helper.h"

#ifndef output
#define output 1
#endif

int main(int argc, char const *argv[])
{
    /* Opencl related variables */
    cl_int            err;
    cl_platform_id    platform = GetPlatform(0); 
    cl_device_id      device = GetDevice(platform, 0);
    cl_context        context;
    cl_command_queue  commands;
    cl_program        p_get_max;
    cl_kernel         k_get_max;
    cl_mem            d_data, d_maxvalue;

    /* Data for pde solver */
    int mtot = 10;
    float data[mtot];
    float max;
    for (int i = 0; i < mtot; ++i)
    {
        data[i] = i;
    }

    std::size_t global=mtot;
    std::size_t local =mtot;

    /* Create context */
    context = clCreateContext(0, 1, &device, NULL, NULL, &err);
    CheckError(err);

    /* Create commands */
    commands = clCreateCommandQueue (context, device, 0, &err);
    CheckError(err);

    /* Create program, kernel from source */

    p_get_max = CreateProgram(LoadKernel ("get_max.cl"), context);
    err     = clBuildProgram(p_get_max, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        ProgramErrMsg(p_get_max, device);
    }
    k_get_max = clCreateKernel(p_get_max, "get_max", &err);

    /* Allocate device memory */
    d_data      = clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(float)*mtot   , NULL, NULL);
    d_maxvalue  = clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(float)        , NULL, NULL);
    CheckError(clEnqueueWriteBuffer(commands, d_data, CL_TRUE, 0, sizeof(float)*mtot, data, 0, NULL, NULL )); 
    /* Set arguments */
    /* Calculate cfl */ 
    CheckError(clSetKernelArg(k_get_max, 0, sizeof(cl_mem), &d_data));
    CheckError(clSetKernelArg(k_get_max, 1, sizeof(cl_mem), &d_maxvalue));
    CheckError(clSetKernelArg(k_get_max, 2, sizeof(float)*local, NULL));

    CheckError(clEnqueueNDRangeKernel(commands, k_get_max, 1, NULL, &global, &local, 0, NULL, NULL));
    CheckError(clEnqueueReadBuffer(commands, d_data, CL_TRUE, 0, sizeof(float)*mtot, data, 0, NULL, NULL ));  
    CheckError(clEnqueueReadBuffer(commands, d_maxvalue, CL_TRUE, 0, sizeof(float), &max, 0, NULL, NULL ));
#if output
    std::cout<<"max = "<<max<<std::endl;
    for (int i = 0; i < mtot; ++i)
    {
            std::cout<<"data["<<std::setw(2)<<i<<"] = "<<std::setw(5)<<data[i]<<std::endl;
    }
#endif

    clReleaseMemObject(d_data);
    clReleaseCommandQueue (commands);
    clReleaseKernel(k_get_max);
    clReleaseProgram(p_get_max);
    clReleaseContext(context);

    return 0;
}