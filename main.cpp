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
    cl_program        p_acoustic_1d, p_qinit, p_bc1;
    cl_kernel         k_acoustic_1d, k_qinit, k_bc1;
    cl_mem            d_p, d_u, d_p_old, d_u_old;

    /* Data for pde solver */
    int meqn = 2;
    int ndim = 1;
    int maux = 0;
    int mx = 10, mbc = 2;
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
    
    /* data */
    float p[mtot], u[mtot];

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
    d_p      = clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(float)*mtot, NULL, NULL);
    d_u      = clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(float)*mtot, NULL, NULL);
    d_p_old  = clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(float)*mtot, NULL, NULL);
    d_u_old  = clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(float)*mtot, NULL, NULL);

    assert(d_p != 0);
    assert(d_u != 0);
    assert(d_p_old != 0);
    assert(d_u_old != 0);

    /* Set arguments */
    /* QINIT */
    CheckError(clSetKernelArg(k_qinit, 0, sizeof(cl_mem), &d_p));
    CheckError(clSetKernelArg(k_qinit, 1, sizeof(cl_mem), &d_u));
    CheckError(clSetKernelArg(k_qinit, 2, sizeof(int)   , &mx));
    CheckError(clSetKernelArg(k_qinit, 3, sizeof(int)   , &mbc));
    CheckError(clSetKernelArg(k_qinit, 4, sizeof(float) , &xlower));
    CheckError(clSetKernelArg(k_qinit, 5, sizeof(float) , &dx));

    /* BC1 */
    CheckError(clSetKernelArg(k_bc1, 0, sizeof(cl_mem), &d_p));
    CheckError(clSetKernelArg(k_bc1, 1, sizeof(cl_mem), &d_u));
    CheckError(clSetKernelArg(k_bc1, 2, sizeof(int)   , &mx));
    CheckError(clSetKernelArg(k_bc1, 3, sizeof(int)   , &mbc));

    /* ADVANCE SOLUTION */
    CheckError(clSetKernelArg(k_acoustic_1d, 0, sizeof(cl_mem), &d_p));
    CheckError(clSetKernelArg(k_acoustic_1d, 1, sizeof(cl_mem), &d_u));
    CheckError(clSetKernelArg(k_acoustic_1d, 2, sizeof(int)   , &mx));
    CheckError(clSetKernelArg(k_acoustic_1d, 3, sizeof(int)   , &mbc));
    CheckError(clSetKernelArg(k_acoustic_1d, 4, sizeof(float) , &dt));
    CheckError(clSetKernelArg(k_acoustic_1d, 5, sizeof(float) , &dx));
    CheckError(clSetKernelArg(k_acoustic_1d, 6, sizeof(float) , &rho));
    CheckError(clSetKernelArg(k_acoustic_1d, 7, sizeof(float) , &K));


    CheckError(clEnqueueNDRangeKernel(commands, k_qinit, 1, NULL, &global, &local, 0, NULL, NULL));

    CheckError(clEnqueueReadBuffer(commands, d_p, CL_TRUE, 0, sizeof(float)*mtot, p, 0, NULL, NULL ));  
    CheckError(clEnqueueReadBuffer(commands, d_u, CL_TRUE, 0, sizeof(float)*mtot, u, 0, NULL, NULL ));
#if 1
    for (int i = 0; i < mtot; ++i)
    {
            std::cout<<"p["<<i<<"] = "<<p[i]<<std::endl;
    }
#endif
    out1(meqn, mbc, mx, xlower, dx, p, u, 0.0, iframe, NULL, maux);
    iframe++;

    /* Launch kernel */
    for (int j = 0; j < 10; ++j)
    {
        t_old = t;
        if (t_old+dt > t_final && t_start < t_final) 
            dt = t_final - t_old;
        t = t_old + dt;
        clEnqueueCopyBuffer (commands, d_p, d_p_old, 0, 0, sizeof(float)*mtot, 0, NULL, NULL);
        clEnqueueCopyBuffer (commands, d_u, d_u_old, 0, 0, sizeof(float)*mtot, 0, NULL, NULL);

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
        if (cfl <= cflmax){
            // Accept this step
            cflmax = std::max(cfl, cflmax);
        }else{
            // Reject this step => Take a smaller step

        }
        
        /* Read ouput array */
        CheckError(clEnqueueReadBuffer(commands, d_p, CL_TRUE, 0, sizeof(float)*mtot, p, 0, NULL, NULL ));  
        CheckError(clEnqueueReadBuffer(commands, d_u, CL_TRUE, 0, sizeof(float)*mtot, u, 0, NULL, NULL ));
#if 1
        std::cout<<std::endl;
        for (int i = 0; i < mtot; ++i)
        {
            std::cout<<"p["<<i<<"] = "<<p[i]<<std::endl;
        }
#endif
        out1(meqn, mbc, mx, xlower, dx, p, u, t, iframe, NULL, maux);
        iframe++;
    }
    
    clReleaseMemObject(d_u);
    clReleaseMemObject(d_p);
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