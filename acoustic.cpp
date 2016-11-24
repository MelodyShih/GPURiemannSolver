#include <iostream>
#include <iomanip>
#include <cmath>
#include <cassert>
#include <fstream>

//#include "OpenCL/opencl.h"
#include <CL/cl.h>
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
    cl_program        p_acoustic_1d, p_qinit, p_bc1, p_max_speed;
    cl_kernel         k_acoustic_1d, k_qinit, k_bc1, k_max_speed;
    cl_mem            d_q, d_q_old, d_s;

    /* Data for pde solver */
    int meqn = 2, mwaves = 2, maux = 0;
    int ndim = 1;
    int mx = 100, mbc = 2, mtot = mx + 2*mbc;
    int nout = 16;
    int iframe = 0;
    
    /* physical domain */
    double xlower = -1.0, xupper = 1.0;
    double dx = (xupper - xlower)/mx;
    
    /* time */
    int maxtimestep = 1000;
    double t = 0, t_old;
    double t_start = 0, t_final = 1.0;
    double dt = dx/2, dtmax = 1.0, dtmin = 0.0;
    double dtout = t_final/nout, tout = 0;
    
    /* data */
    double q[meqn*mtot];

    /* problem data */
    double K = 4.0, rho = 1.0;
    
    /* other */
    double cfl, cflmax = 1.0, cfldesire = 0.9;
    char outdir[] = "Output/acoustic";

    std::size_t local    = mtot/2;
    std::size_t numgroup = ((mtot - 1)/local + 1);
    std::size_t global   = numgroup * local;
    std::size_t l;

    cl_device_fp_config cfg;
    clGetDeviceInfo(device, CL_DEVICE_DOUBLE_FP_CONFIG, sizeof(cfg), &cfg, NULL);
    printf("Double FP config = %llu\n", cfg);

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


    p_qinit = CreateProgram(LoadKernel ("Kernel/acoustic_qinit.cl"), context);
    err     = clBuildProgram(p_qinit, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        ProgramErrMsg(p_qinit, device);
    }
    k_qinit = clCreateKernel(p_qinit, "acoustic_qinit", &err);
    CheckError(err);

    p_bc1 = CreateProgram(LoadKernel ("Kernel/bc1.cl"), context);
    err     = clBuildProgram(p_bc1, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        ProgramErrMsg(p_bc1, device);
    }
    k_bc1 = clCreateKernel(p_bc1, "bc1", &err);
    CheckError(err);

    p_max_speed = CreateProgram(LoadKernel ("Kernel/max_speed.cl"), context);
    err     = clBuildProgram(p_max_speed, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        ProgramErrMsg(p_max_speed, device);
    }
    k_max_speed = clCreateKernel(p_max_speed, "max_speed", &err);

    /* Allocate device memory */
    d_q      = clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(double)*meqn*mtot, NULL, NULL);
    d_q_old  = clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(double)*meqn*mtot, NULL, NULL);
    d_s      = clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(double)*mtot     , NULL, NULL);

    /* Set arguments */
    /* QINIT */
    CheckError(clSetKernelArg(k_qinit, 0, sizeof(cl_mem), &d_q));
    CheckError(clSetKernelArg(k_qinit, 1, sizeof(int)   , &meqn));
    CheckError(clSetKernelArg(k_qinit, 2, sizeof(int)   , &mx));
    CheckError(clSetKernelArg(k_qinit, 3, sizeof(int)   , &mbc));
    CheckError(clSetKernelArg(k_qinit, 4, sizeof(double) , &xlower));
    CheckError(clSetKernelArg(k_qinit, 5, sizeof(double) , &dx));

    /* BC1 */
    CheckError(clSetKernelArg(k_bc1, 0, sizeof(cl_mem), &d_q));
    CheckError(clSetKernelArg(k_bc1, 1, sizeof(int)   , &meqn));
    CheckError(clSetKernelArg(k_bc1, 2, sizeof(int)   , &mx));
    CheckError(clSetKernelArg(k_bc1, 3, sizeof(int)   , &mbc));

    /* ADVANCE SOLUTION */
    CheckError(clSetKernelArg(k_acoustic_1d, 0, sizeof(cl_mem), &d_q));
    CheckError(clSetKernelArg(k_acoustic_1d, 1, sizeof(cl_mem), &d_s));
    CheckError(clSetKernelArg(k_acoustic_1d, 2, sizeof(int)   , &mx));
    CheckError(clSetKernelArg(k_acoustic_1d, 3, sizeof(int)   , &mbc));
    CheckError(clSetKernelArg(k_acoustic_1d, 4, sizeof(double) , &dt));
    CheckError(clSetKernelArg(k_acoustic_1d, 5, sizeof(double) , &dx));
    CheckError(clSetKernelArg(k_acoustic_1d, 6, sizeof(double) , &rho));
    CheckError(clSetKernelArg(k_acoustic_1d, 7, sizeof(double) , &K));

    /* Calculate cfl */ 
    CheckError(clSetKernelArg(k_max_speed, 0, sizeof(cl_mem), &d_s));
    CheckError(clSetKernelArg(k_max_speed, 1, sizeof(double)*local, NULL));

    CheckError(clEnqueueNDRangeKernel(commands, k_qinit, 1, NULL, &global, &local, 0, NULL, NULL));
    CheckError(clEnqueueReadBuffer(commands, d_q, CL_TRUE, 0, sizeof(double)*mtot*meqn, q, 0, NULL, NULL ));  
#if output
    for (int i = 0; i < mtot; ++i)
    {
            std::cout<<"p["<<std::setw(2)<<i<<"] = "<<std::setw(5)<<q[2*i]
                     <<",  u["<<std::setw(2)<<i<<"] = "<<std::setw(5)<<q[2*i+1]<<std::endl;
    }
#endif

    out1(meqn, mbc, mx, xlower, dx, q, 0.0, iframe, NULL, maux, outdir);
    iframe++;

    /* Launch kernel */
    tout += dtout;
    for (int j = 0; j < maxtimestep; ++j)
    {
        t_old = t;
        if (t_old+dt > t_final && t_start < t_final) 
            dt = t_final - t_old;
        t = t_old + dt;
        clEnqueueCopyBuffer (commands, d_q, d_q_old, 0, 0, sizeof(double)*mtot*meqn, 0, NULL, NULL);

        CheckError(clEnqueueNDRangeKernel(commands, k_bc1, 1, NULL, &global, &local, 0, NULL, NULL));
        CheckError(clEnqueueNDRangeKernel(commands, k_acoustic_1d, 1, NULL, &global, &local, 0, NULL, NULL));

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
        std::cout<<"At time "<<t<<" cfl = "<<cfl<<std::endl;
        /* Choose new time step if variable time step */
        if (cfl > 0){
            dt = std::min(dtmax, dt*cfldesire/cfl);
            // dtmin = std::min(dt,dtmin);
            // dtmax = std::max(dt,dtmax);
        }else{
            dt = dtmax;
        }
        CheckError(clSetKernelArg(k_acoustic_1d, 4, sizeof(double) , &dt));
        
        /* Check to see if the Courant number was too large */
        if (cfl <= cflmax){
            //Accept this step
            //cflmax = std::max(cfl, cflmax);
        }else{
            // Reject this step => Take a smaller step
            std::cout<<"-----Reject this step-----"<<std::endl;
            clEnqueueCopyBuffer (commands, d_q_old, d_q, 0, 0, sizeof(double)*mtot*meqn, 0, NULL, NULL);
            t = t_old;
        }

        /* Read ouput array */
        CheckError(clEnqueueReadBuffer(commands, d_q, CL_TRUE, 0, sizeof(double)*mtot*meqn, q, 0, NULL, NULL ));  
#if output
        std::cout<<std::endl;
        for (int i = mbc; i < mtot - mbc; ++i)
        {
                std::cout<<"p["<<std::setw(2)<<i<<"] = "<<std::setw(5)<<q[2*i]
                         <<",  u["<<std::setw(2)<<i<<"] = "<<std::setw(5)<<q[2*i+1]<<std::endl;
        }
#endif
        if (t >= tout)
        {
            out1(meqn, mbc, mx, xlower, dx, q, t, iframe, NULL, maux, outdir);
            iframe++;
            tout += dtout;
        }
        if (t >= t_final)    break;
    }
    
    clReleaseMemObject(d_q);
    clReleaseMemObject(d_s);
    clReleaseCommandQueue (commands);
    clReleaseKernel(k_acoustic_1d);
    clReleaseKernel(k_bc1);
    clReleaseKernel(k_qinit);
    clReleaseKernel(k_max_speed);
    clReleaseProgram(p_acoustic_1d);
    clReleaseProgram(p_bc1);
    clReleaseProgram(p_qinit);
    clReleaseProgram(p_max_speed);
    clReleaseContext(context);
    return 0;
}
