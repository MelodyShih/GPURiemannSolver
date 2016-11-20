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
    cl_program        p_rp1_acoustic, p_qinit, p_bc1, 
                      p_update_q1, p_calc_cfl;
    cl_kernel         k_rp1_acoustic, k_qinit, k_bc1, 
                      k_update_q1, k_calc_cfl;
    cl_mem            d_q, d_q_old, d_apdq, d_amdq, d_s, d_cfl;

    /* Data for pde solver */
    int meqn = 2, mwaves = 2;
    int ndim = 1;
    int maux = 0;
    int mx = 100, mbc = 2;
    int mtot = mx + 2*mbc;
    int nout = 10;
    int iframe = 0;
    
    /* physical domain */
    float xlower = -1.0;
    float xupper = 1.0;
    float dx = (xupper - xlower)/mx;
    
    /* time */
    int maxtimestep = 50;
    float t = 0;
    float t_old;
    float t_start = 0, t_final = 1.0;
    float dt = dx / 2;
    float dtmax = 1.0, dtmin = 0.0;
    
    /* data */
    float q[meqn*mtot];
    float s[mx+1];

    /* problem data */
    float K = 4.0, rho = 1.0;
    
    /* other */
    float cfl = 1;
    float cflmax = 1;
    float cfldesire = 1;

    std::size_t global=mtot;
    std::size_t local =mtot/2;

    /* Create context */
    context = clCreateContext(0, 1, &device, NULL, NULL, &err);
    CheckError(err);

    /* Create commands */
    commands = clCreateCommandQueue (context, device, 0, &err);
    CheckError(err);

    /* Create program, kernel from source */
    p_rp1_acoustic = CreateProgram(LoadKernel ("Kernel/rp1_acoustic.cl"), context);
    err     = clBuildProgram(p_rp1_acoustic, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        ProgramErrMsg(p_rp1_acoustic, device);
    }
    k_rp1_acoustic = clCreateKernel(p_rp1_acoustic, "rp1_acoustic", &err);

    p_update_q1 = CreateProgram(LoadKernel ("Kernel/update_q1.cl"), context);
    err     = clBuildProgram(p_update_q1, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        ProgramErrMsg(p_update_q1, device);
    }
    k_update_q1 = clCreateKernel(p_update_q1, "update_q1", &err);

    p_calc_cfl = CreateProgram(LoadKernel ("Kernel/calc_cfl.cl"), context);
    err     = clBuildProgram(p_calc_cfl, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        ProgramErrMsg(p_calc_cfl, device);
    }
    k_calc_cfl = clCreateKernel(p_calc_cfl, "calc_cfl", &err);


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
    d_apdq   = clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(float)*meqn*mtot, NULL, NULL);
    d_amdq   = clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(float)*meqn*mtot, NULL, NULL);
    d_s      = clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(float)*(mx+1)   , NULL, NULL);
    d_cfl    = clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(float)          , NULL, NULL);

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

    /* BC1 */
    CheckError(clSetKernelArg(k_bc1, 0, sizeof(cl_mem), &d_q));
    CheckError(clSetKernelArg(k_bc1, 1, sizeof(int)   , &meqn));
    CheckError(clSetKernelArg(k_bc1, 2, sizeof(int)   , &mx));
    CheckError(clSetKernelArg(k_bc1, 3, sizeof(int)   , &mbc));

    /* RP1 */
    CheckError(clSetKernelArg(k_rp1_acoustic, 0, sizeof(cl_mem), &d_q));
    CheckError(clSetKernelArg(k_rp1_acoustic, 1, sizeof(cl_mem), &d_apdq));
    CheckError(clSetKernelArg(k_rp1_acoustic, 2, sizeof(cl_mem), &d_amdq));
    CheckError(clSetKernelArg(k_rp1_acoustic, 3, sizeof(cl_mem), &d_s));
    CheckError(clSetKernelArg(k_rp1_acoustic, 4, sizeof(int)   , &mx));
    CheckError(clSetKernelArg(k_rp1_acoustic, 5, sizeof(int)   , &mbc));
    CheckError(clSetKernelArg(k_rp1_acoustic, 6, sizeof(float) , &rho));
    CheckError(clSetKernelArg(k_rp1_acoustic, 7, sizeof(float) , &K));

    /* UPDATE_Q */
    CheckError(clSetKernelArg(k_update_q1, 0, sizeof(cl_mem), &d_q));
    CheckError(clSetKernelArg(k_update_q1, 1, sizeof(cl_mem), &d_apdq));
    CheckError(clSetKernelArg(k_update_q1, 2, sizeof(cl_mem), &d_amdq));
    CheckError(clSetKernelArg(k_update_q1, 3, sizeof(int)   , &meqn));
    CheckError(clSetKernelArg(k_update_q1, 4, sizeof(int)   , &mx));
    CheckError(clSetKernelArg(k_update_q1, 5, sizeof(int)   , &mbc));
    CheckError(clSetKernelArg(k_update_q1, 6, sizeof(float) , &dx));
    CheckError(clSetKernelArg(k_update_q1, 7, sizeof(float) , &dt));

    /* Calculate cfl */ 
    CheckError(clSetKernelArg(k_calc_cfl, 0, sizeof(cl_mem), &d_s));
    CheckError(clSetKernelArg(k_calc_cfl, 1, sizeof(cl_mem), &d_cfl));
    CheckError(clSetKernelArg(k_calc_cfl, 2, sizeof(float) , &dx));
    CheckError(clSetKernelArg(k_calc_cfl, 3, sizeof(float) , &dt));
    CheckError(clSetKernelArg(k_calc_cfl, 4, sizeof(int), &mx));

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
    for (int j = 0; j < maxtimestep; ++j)
    {
        t_old = t;
        if (t_old+dt > t_final && t_start < t_final) 
            dt = t_final - t_old;
        t = t_old + dt;
        printf("t = %f, dt = %f\n", t, dt);
        clEnqueueCopyBuffer (commands, d_q, d_q_old, 0, 0, sizeof(float)*mtot*meqn, 0, NULL, NULL);

        CheckError(clEnqueueNDRangeKernel(commands, k_bc1, 1, NULL, &global, &local, 0, NULL, NULL));
        CheckError(clEnqueueNDRangeKernel(commands, k_rp1_acoustic, 1, NULL, &global, &local, 0, NULL, NULL));
        CheckError(clEnqueueNDRangeKernel(commands, k_update_q1, 1, NULL, &global, &local, 0, NULL, NULL));
        CheckError(clEnqueueNDRangeKernel(commands, k_calc_cfl, 1, NULL, &global, &local, 0, NULL, NULL));
        
        CheckError(clEnqueueReadBuffer(commands, d_cfl, CL_TRUE, 0, sizeof(float), &cfl, 0, NULL, NULL));
        printf("cfl = %f\n", cfl);
        /* Choose new time step if variable time step */
        if (cfl > 0){
            dt = std::min(dtmax, dt*cfldesire/cfl);
            printf("dt = %f\n", dt);
            // dtmin = std::min(dt,dtmin);
            // dtmax = std::max(dt,dtmax);
        }else{
            dt = dtmax;
        }
        CheckError(clSetKernelArg(k_update_q1, 7, sizeof(float) , &dt));
        CheckError(clSetKernelArg(k_calc_cfl, 3, sizeof(float) , &dt));
        // /* Check to see if the Courant number was too large */
        // if (cfl <= cflmax){
        //     // Accept this step
        //     cflmax = std::max(cfl, cflmax);
        // }else{
        //     // Reject this step => Take a smaller step

        // }
        if (t >= t_final)    break;

        /* Read ouput array */
        CheckError(clEnqueueReadBuffer(commands, d_q, CL_TRUE, 0, sizeof(float)*mtot*meqn, q, 0, NULL, NULL));
#if output
        std::cout<<std::endl;
        for (int i = mbc; i < mtot - mbc; ++i)
        {
            std::cout<<"p["<<std::setw(2)<<i<<"] = "<<std::setw(5)<<q[2*i]
                     <<",  u["<<std::setw(2)<<i<<"] = "<<std::setw(5)<<q[2*i+1]<<std::endl;
        }
#endif
        out1(meqn, mbc, mx, xlower, dx, q, t, iframe, NULL, maux);
        iframe++;
    }

    clReleaseMemObject(d_q);
    clReleaseMemObject(d_q_old);
    clReleaseMemObject(d_apdq);
    clReleaseMemObject(d_amdq);
    clReleaseMemObject(d_s);
    clReleaseCommandQueue (commands);
    clReleaseKernel(k_rp1_acoustic);
    clReleaseKernel(k_bc1);
    clReleaseKernel(k_qinit);
    clReleaseKernel(k_update_q1);
    clReleaseKernel(k_calc_cfl);

    clReleaseProgram(p_rp1_acoustic);
    clReleaseProgram(p_bc1);
    clReleaseProgram(p_qinit);
    clReleaseProgram(p_update_q1);
    clReleaseProgram(p_calc_cfl);
    clReleaseContext(context);

    return 0;
}