#include <iostream>
#include <iomanip>
#include <cmath>
#include <cassert>
#include <fstream>
#include <time.h>

#include <CL/cl.h>
#include "helper.h"

#ifndef output
#define output 0
#endif

int main(int argc, char const *argv[])
{
    /* Opencl related variables */
    cl_int            err = 0;
    cl_platform_id    platform = GetPlatform(0); 
    cl_device_id      device   = GetDevice(platform, 0);
    cl_context        context;

    cl_command_queue  commands;
    cl_program        p_rp1_euler, p_qinit, p_bc1, 
                      p_update_q1, p_max_speed;
    cl_kernel         k_rp1_euler, k_qinit, k_bc1, 
                      k_update_q1, k_max_speed;
    cl_mem            d_q, d_q_old, d_apdq, d_amdq, d_s;

    /* Data for pde solver */
    int meqn = 3, mwaves = 3, maux = 0;
    int ndim = 1;
    int mx = 764, mbc = 2, mtot = mx + 2*mbc;
    int nout = 10;
    int iframe = 0;
    
    /* physical domain */
    double xlower = 0.0, xupper = 1.0;
    double dx = (xupper - xlower)/mx;
    
    /* time */
    int maxtimestep = 10000000;
    int totaltimestep;
    double t = 0, t_old;
    double t_start = 0, t_final = 0.038;
    double dt = 0.1, dtmax = 1.0, dtmin = 0.0;
    double dtout = t_final/nout, tout = 0;
    
    /* data */
    double q[meqn*mtot];
    double s[meqn*mtot];
    /* problem data */
    double gamma = 1.4;
    
    /* other */
    double cfl, cflmax = 1, cfldesire = 0.9;
    char outdir[] = "Output/euler";
    
    std::size_t local    = 256;
    std::size_t numgroup = ((mtot - 1)/local + 1);
    std::size_t global   = numgroup * local;
    std::size_t l;

    // std::cout<<"local = "<<local<<"; global = "<<global<<std::endl;
    /* Create context */
    context = clCreateContext(0, 1, &device, NULL, NULL, &err);
    CheckError(err);
    /* Create commands */
    commands = clCreateCommandQueue (context, device, 0, &err);
    CheckError(err);

    /* Create program, kernel from source */
    p_rp1_euler = CreateProgram(LoadKernel ("Kernel/rp1_euler.cl"), context);
    err     = clBuildProgram(p_rp1_euler, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        ProgramErrMsg(p_rp1_euler, device);
    }
    k_rp1_euler = clCreateKernel(p_rp1_euler, "rp1_euler", &err);

    p_update_q1 = CreateProgram(LoadKernel ("Kernel/update_q1.cl"), context);
    err     = clBuildProgram(p_update_q1, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        ProgramErrMsg(p_update_q1, device);
    }
    k_update_q1 = clCreateKernel(p_update_q1, "update_q1", &err);

    p_max_speed = CreateProgram(LoadKernel ("Kernel/max_speed.cl"), context);
    err     = clBuildProgram(p_max_speed, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        ProgramErrMsg(p_max_speed, device);
    }
    k_max_speed = clCreateKernel(p_max_speed, "max_speed", &err);


    p_qinit = CreateProgram(LoadKernel ("Kernel/euler_qinit.cl"), context);
    err     = clBuildProgram(p_qinit, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        ProgramErrMsg(p_qinit, device);
    }
    k_qinit = clCreateKernel(p_qinit, "euler_qinit", &err);
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
    d_q      = clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(double)*meqn*mtot, NULL, NULL);
    d_q_old  = clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(double)*meqn*mtot, NULL, NULL);
    d_apdq   = clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(double)*meqn*mtot, NULL, NULL);
    d_amdq   = clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(double)*meqn*mtot, NULL, NULL);
    d_s      = clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(double)*mtot     , NULL, NULL);

    /* Set arguments */
    /* QINIT */
    CheckError(clSetKernelArg(k_qinit, 0, sizeof(cl_mem), &d_q));
    CheckError(clSetKernelArg(k_qinit, 1, sizeof(int)   , &meqn));
    CheckError(clSetKernelArg(k_qinit, 2, sizeof(int)   , &mx));
    CheckError(clSetKernelArg(k_qinit, 3, sizeof(int)   , &mbc));
    CheckError(clSetKernelArg(k_qinit, 4, sizeof(double) , &xlower));
    CheckError(clSetKernelArg(k_qinit, 5, sizeof(double) , &dx));
    CheckError(clSetKernelArg(k_qinit, 6, sizeof(double) , &gamma));

    /* BC1 */
    CheckError(clSetKernelArg(k_bc1, 0, sizeof(cl_mem), &d_q));
    CheckError(clSetKernelArg(k_bc1, 1, sizeof(int)   , &meqn));
    CheckError(clSetKernelArg(k_bc1, 2, sizeof(int)   , &mx));
    CheckError(clSetKernelArg(k_bc1, 3, sizeof(int)   , &mbc));

    /* RP1 */
    CheckError(clSetKernelArg(k_rp1_euler, 0, sizeof(cl_mem), &d_q));
    CheckError(clSetKernelArg(k_rp1_euler, 1, sizeof(cl_mem), &d_apdq));
    CheckError(clSetKernelArg(k_rp1_euler, 2, sizeof(cl_mem), &d_amdq));
    CheckError(clSetKernelArg(k_rp1_euler, 3, sizeof(cl_mem), &d_s));
    CheckError(clSetKernelArg(k_rp1_euler, 4, sizeof(int)   , &mx));
    CheckError(clSetKernelArg(k_rp1_euler, 5, sizeof(int)   , &mbc));
    CheckError(clSetKernelArg(k_rp1_euler, 6, sizeof(double) , &gamma));

    /* UPDATE_Q */
    CheckError(clSetKernelArg(k_update_q1, 0, sizeof(cl_mem), &d_q));
    CheckError(clSetKernelArg(k_update_q1, 1, sizeof(cl_mem), &d_apdq));
    CheckError(clSetKernelArg(k_update_q1, 2, sizeof(cl_mem), &d_amdq));
    CheckError(clSetKernelArg(k_update_q1, 3, sizeof(int)   , &meqn));
    CheckError(clSetKernelArg(k_update_q1, 4, sizeof(int)   , &mx));
    CheckError(clSetKernelArg(k_update_q1, 5, sizeof(int)   , &mbc));
    CheckError(clSetKernelArg(k_update_q1, 6, sizeof(double) , &dx));
    CheckError(clSetKernelArg(k_update_q1, 7, sizeof(double) , &dt));

    /* Calculate cfl */ 
    CheckError(clSetKernelArg(k_max_speed, 0, sizeof(cl_mem), &d_s));
    CheckError(clSetKernelArg(k_max_speed, 1, sizeof(double)*local, NULL));

    CheckError(clEnqueueNDRangeKernel(commands, k_qinit, 1, NULL, &global, &local, 0, NULL, NULL));
    CheckError(clEnqueueReadBuffer(commands, d_q, CL_TRUE, 0, sizeof(double)*mtot*meqn, q, 0, NULL, NULL ));  

#if output
    for (int i = 0; i < mtot; ++i)
    {
            std::cout<<"rho["<<std::setw(2)<<i<<"] = "<<std::setw(5)<<q[3*i]
                     <<",  rhou["<<std::setw(2)<<i<<"] = "<<std::setw(5)<<q[3*i+1]
                     <<",     e["<<std::setw(2)<<i<<"] = "<<std::setw(5)<<q[3*i+2]
                     <<std::endl;
    }
#endif

    out1(meqn, mbc, mx, xlower, dx, q, 0.0, iframe, NULL, maux, outdir);
    iframe++;
    clock_t begin = clock();
    tout += dtout;
    /* Launch kernel */
    for (int j = 0; j < maxtimestep; ++j)
    {
        t_old = t;
        if (t_old+dt > t_final && t_start < t_final) 
            dt = t_final - t_old;
        t = t_old + dt;
        clEnqueueCopyBuffer (commands, d_q, d_q_old, 0, 0, sizeof(double)*mtot*meqn, 0, NULL, NULL);

        CheckError(clEnqueueNDRangeKernel(commands, k_bc1, 1, NULL, &global, &local, 0, NULL, NULL));
        CheckError(clEnqueueNDRangeKernel(commands, k_rp1_euler, 1, NULL, &global, &local, 0, NULL, NULL));
        CheckError(clEnqueueNDRangeKernel(commands, k_update_q1, 1, NULL, &global, &local, 0, NULL, NULL));
#if output        
        CheckError(clEnqueueReadBuffer(commands, d_s, CL_TRUE, 0, sizeof(double)*mtot, &s, 0, NULL, NULL ));
        for (int i = 0; i < mtot; ++i)
        {
            std::cout<<s[i]<<" "<<std::endl;
        }
#endif
        /* Get the maximum speed by recursively calling k_max_speed kernel */ 
        l = local;
        for (std::size_t length = global; length > 1; length = numgroup)
        {
            CheckError(clEnqueueNDRangeKernel(commands, k_max_speed, 1, NULL, &length, &l, 0, NULL, NULL));
            numgroup = (length - 1)/l + 1;
            if (numgroup < l) l = numgroup;
        }
        CheckError(clEnqueueReadBuffer(commands, d_s, CL_TRUE, 0, sizeof(double), &cfl, 0, NULL, NULL ));
        cfl *= dt/dx;
        // std::cout<<"At time "<<std::setw(10)<<t<<" cfl = "<<std::setw(10)<<cfl<<std::endl;
        /* Choose new time step if variable time step */
        if (cfl > 0){
            dt = std::min(dtmax, dt*cfldesire/cfl);
            dtmin = std::min(dt,dtmin);
            dtmax = std::max(dt,dtmax);
        }else{
            dt = dtmax;
        }
        CheckError(clSetKernelArg(k_update_q1, 7, sizeof(double) , &dt));
        /* Check to see if the Courant number was too large */
        if (cfl <= cflmax){
            // Accept this step
            //cflmax = std::max(cfl, cflmax);
        }else{
            // Reject this step => Take a smaller step
            std::cout<<"-----Reject this step-----"<<std::endl;
            clEnqueueCopyBuffer (commands, d_q_old, d_q, 0, 0, sizeof(double)*mtot*meqn, 0, NULL, NULL);
            t = t_old;
        }
#if output
        CheckError(clEnqueueReadBuffer(commands, d_q, CL_TRUE, 0, sizeof(double)*mtot*meqn, q, 0, NULL, NULL));
        for (int i = 0; i < mtot; ++i)
        {
                std::cout<<"rho["<<std::setw(2)<<i<<"] = "<<std::setw(5)<<q[3*i]
                         <<",  rhou["<<std::setw(2)<<i<<"] = "<<std::setw(5)<<q[3*i+1]
                         <<",     e["<<std::setw(2)<<i<<"] = "<<std::setw(5)<<q[3*i+2]
                         <<std::endl;
        }
#endif
        if (t >= tout)
        {
            std::cout<<"INFO: Solution "<<iframe<<" computed for time "<<t<<std::endl;
            CheckError(clEnqueueReadBuffer(commands, d_q, CL_TRUE, 0, sizeof(double)*mtot*meqn, q, 0, NULL, NULL));
            out1(meqn, mbc, mx, xlower, dx, q, t, iframe, NULL, maux, outdir);
            iframe++;
            tout += dtout;
        }
        if (t >= t_final){
            std::cout<<"Total time steps = "<<j<<std::endl;
            totaltimestep = j;
            break;
        }
    }
    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    std::cout<<"Total execution time (secs) = "<<time_spent<<std::endl;
    std::cout<<"Time/step = "<<time_spent/totaltimestep<<std::endl;

    clReleaseMemObject(d_q);
    clReleaseMemObject(d_q_old);
    clReleaseMemObject(d_apdq);
    clReleaseMemObject(d_amdq);
    clReleaseMemObject(d_s);
    clReleaseCommandQueue (commands);
    
    clReleaseKernel(k_rp1_euler);
    clReleaseKernel(k_bc1);
    clReleaseKernel(k_qinit);
    clReleaseKernel(k_update_q1);
    clReleaseKernel(k_max_speed);

    clReleaseProgram(p_rp1_euler);
    clReleaseProgram(p_bc1);
    clReleaseProgram(p_qinit);
    clReleaseProgram(p_update_q1);
    clReleaseProgram(p_max_speed);
    clReleaseContext(context);
    clRetainDevice(device);
    return 0;
}