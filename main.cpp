#include <iostream>
#include <iomanip>
#include <cmath>
#include <cassert>
#include <fstream>

#include "OpenCL/opencl.h"
#include "helper.h"

#ifndef output
#define output 0
#endif

int main(int argc, char const *argv[])
{
    /* Opencl related variables */
    cl_int            err;
    cl_platform_id    platform = GetPlatform(0); 
    cl_device_id      device = GetDevice(platform, 0);
    cl_context        context;
    cl_command_queue  commands;
    cl_program        p_acoustic_1d, p_qinit, p_bc1;
    cl_kernel         k_acoustic_1d, k_qinit, k_bc1;
    cl_mem            d_q, d_q_old;

    /* Data for pde solver */
    int meqn = 2, mwaves = 2;
    int ndim = 1;
    int maux = 0;
    int mx = 10, mbc = 1;
    int mtot = mx + 2*mbc;
    int nout = 10;
    int iframe = 0;
    
    /* physical domain */
    float xlower = -1.0;
    float xupper = 1.0;
    float dx = (xupper - xlower)/mx;
    
    /* time */
    int maxtimestep = 200;
    float t = 0;
    float t_old;
    float t_start = 0;
    float t_final = 0.038;
    float dt = dx / 2;
    float dtmax = 0.1;
    float dtmin = 0;
    float dtout = t_final/nout;
    float tout = 0;
    
    /* data */
    float q[meqn*mtot];

    /* problem data */
    float K = 4.0, rho = 1.0;
    /* other */
    float cfl = 1;
    float cflmax = 1;
    float cfldesire = 0.9;

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

    p_bc1 = CreateProgram(LoadKernel ("Kernel/bc1.cl"), context);
    err     = clBuildProgram(p_bc1, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        ProgramErrMsg(p_bc1, device);
    }
    k_bc1 = clCreateKernel(p_bc1, "bc1", &err);
    CheckError(err);

    /* Allocate device memory */
    d_q      = clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(float)*meqn*mtot, NULL, NULL);
    d_q_old  = clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(float)*meqn*mtot, NULL, NULL);

    assert(d_q != 0);
    assert(d_q_old != 0);

    /* Set arguments */
    /* QINIT */
    CheckError(clSetKernelArg(k_qinit, 0, sizeof(cl_mem), &d_q));
    CheckError(clSetKernelArg(k_qinit, 1, sizeof(int)   , &meqn));
    CheckError(clSetKernelArg(k_qinit, 2, sizeof(int)   , &mx));
    CheckError(clSetKernelArg(k_qinit, 3, sizeof(int)   , &mbc));
    CheckError(clSetKernelArg(k_qinit, 4, sizeof(float) , &xlower));
    CheckError(clSetKernelArg(k_qinit, 5, sizeof(float) , &dx));

    // /* BC1 */
    CheckError(clSetKernelArg(k_bc1, 0, sizeof(cl_mem), &d_q));
    CheckError(clSetKernelArg(k_bc1, 1, sizeof(int)   , &meqn));
    CheckError(clSetKernelArg(k_bc1, 2, sizeof(int)   , &mx));
    CheckError(clSetKernelArg(k_bc1, 3, sizeof(int)   , &mbc));

    // /* ADVANCE SOLUTION */
    CheckError(clSetKernelArg(k_acoustic_1d, 0, sizeof(cl_mem), &d_q));
    CheckError(clSetKernelArg(k_acoustic_1d, 1, sizeof(int)   , &mx));
    CheckError(clSetKernelArg(k_acoustic_1d, 2, sizeof(int)   , &mbc));
    CheckError(clSetKernelArg(k_acoustic_1d, 3, sizeof(float) , &dt));
    CheckError(clSetKernelArg(k_acoustic_1d, 4, sizeof(float) , &dx));
    CheckError(clSetKernelArg(k_acoustic_1d, 5, sizeof(float) , &rho));
    CheckError(clSetKernelArg(k_acoustic_1d, 6, sizeof(float) , &K));


    CheckError(clEnqueueNDRangeKernel(commands, k_qinit, 1, NULL, &global, &local, 0, NULL, NULL));

    CheckError(clEnqueueReadBuffer(commands, d_q, CL_TRUE, 0, sizeof(float)*mtot*meqn, q, 0, NULL, NULL ));  
#if output
    for (int i = 0; i < mtot; ++i)
    {
            std::cout<<"p["<<std::setw(2)<<i<<"] = "<<std::setw(5)<<q[2*i]
                     <<",  u["<<std::setw(2)<<i<<"] = "<<std::setw(5)<<q[2*i+1]<<std::endl;
    }
#endif

    out1(meqn, mbc, mx, xlower, dx, q, 0.0, iframe, NULL, maux);
    iframe++;

    /* Launch kernel */
    for (int j = 0; j < 10; ++j)
    {
        t_old = t;
        if (t_old+dt > t_final && t_start < t_final) 
            dt = t_final - t_old;
        t = t_old + dt;
        clEnqueueCopyBuffer (commands, d_q, d_q_old, 0, 0, sizeof(float)*mtot*meqn, 0, NULL, NULL);

        CheckError(clEnqueueNDRangeKernel(commands, k_bc1, 1, NULL, &global, &local, 0, NULL, NULL));
        CheckError(clEnqueueNDRangeKernel(commands, k_acoustic_1d, 1, NULL, &global, &local, 0, NULL, NULL));
        
        /* Choose new time step if variable time step */
        if (cfl > 0){
            dt = std::min(dtmax, dt*cfldesire/cfl);
            dtmin = std::min(dt,dtmin);
            dtmax = std::max(dt,dtmax);
        }else{
            dt = dtmax;
        }
        
        /* Check to see if the Courant number was too large */
        // if (cfl <= cflmax){
        //     // Accept this step
        //     cflmax = std::max(cfl, cflmax);
        // }else{
        //     // Reject this step => Take a smaller step

        // }
        
        /* Read ouput array */
        CheckError(clEnqueueReadBuffer(commands, d_q, CL_TRUE, 0, sizeof(float)*mtot*meqn, q, 0, NULL, NULL ));  
#if output
        std::cout<<std::endl;
        for (int i = mbc; i < mtot - mbc; ++i)
        {
                std::cout<<"p["<<std::setw(2)<<i<<"] = "<<std::setw(5)<<q[2*i]
                         <<",  u["<<std::setw(2)<<i<<"] = "<<std::setw(5)<<q[2*i+1]<<std::endl;
        }
#endif
        if (t > tout)
        {
            out1(meqn, mbc, mx, xlower, dx, q, t, iframe, NULL, maux);
            iframe++;
            tout += dtout;
        }
    }
    
    clReleaseMemObject(d_q);
    clReleaseCommandQueue (commands);
    clReleaseKernel(k_acoustic_1d);
    clReleaseKernel(k_bc1);
    clReleaseKernel(k_qinit);
    clReleaseProgram(p_acoustic_1d);
    clReleaseProgram(p_bc1);
    clReleaseProgram(p_qinit);
    clReleaseContext(context);

    return 0;
}