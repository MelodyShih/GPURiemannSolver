#include <iostream>
#include <iomanip>
#include <cmath>
#include <cassert>
#include <fstream>

#include "OpenCL/opencl.h"
#include "../helper.h"

#ifndef output
#define output 0
#endif

int main(int argc, char const *argv[])
{
    const char* version;
    if (argv[1] == NULL) 
        version = "get_max.cl";
    else 
        version = argv[1];
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
    int mtot = 256;
    float data[mtot];
    float max;
    for (int i = 0; i < mtot; ++i)
    {
        data[i] = i+ 1;
    }

    std::size_t local = 128;
    std::size_t global= ((mtot - 1)/local + 1) * local;
    /* Create context */
    context = clCreateContext(0, 1, &device, NULL, NULL, &err);
    CheckError(err);

    /* Create commands */
    commands = clCreateCommandQueue (context, device, 0, &err);
    CheckError(err);

    /* Create program, kernel from source */

    p_get_max = CreateProgram(LoadKernel (version), context);
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
    CheckError(clSetKernelArg(k_get_max, 1, sizeof(float)*local, NULL));
    CheckError(clSetKernelArg(k_get_max, 2, sizeof(int), &mtot));

    std::size_t numgroup;
    for (std::size_t length = global; length > 1; length = numgroup)
    {
        //numgroup = ((length - 1)/local + 1);
        std::cout<<std::endl;
        std::cout<<"length = "<<length<<std::endl;
        //global = numgroup * local;
        CheckError(clEnqueueNDRangeKernel(commands, k_get_max, 1, NULL, &length, &local, 0, NULL, NULL));
        numgroup = (length - 1)/local + 1;
        if (numgroup < local)
        {
            local  = numgroup;
        }
        std::cout<<"group = "<<numgroup<<std::endl;
        CheckError(clEnqueueReadBuffer(commands, d_data, CL_TRUE, 0, sizeof(float)*numgroup, data, 0, NULL, NULL ));
        for (int i = 0; i < numgroup; ++i)
        {
            std::cout<<"data["<<std::setw(2)<<i<<"] = "<<std::setw(5)<<data[i]<<std::endl;
        }
    }
    // CheckError(clEnqueueReadBuffer(commands, d_data, CL_TRUE, 0, sizeof(float)*mtot, data, 0, NULL, NULL ));  
    // CheckError(clEnqueueReadBuffer(commands, d_maxvalue, CL_TRUE, 0, sizeof(float), &max, 0, NULL, NULL ));
#if output
    std::cout<<"max = "<<data[0]<<std::endl;
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