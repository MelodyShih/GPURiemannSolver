#include <iostream>
#include <iomanip>
#include <cmath>
#include <cassert>
#include <fstream>
#include <time.h>

#include <CL/cl.h>
#include "helper.h"

int main(int argc, char const *argv[])
{
    // -------------------------------------------------------------------------------//
    //    This code is used to get the information of how many registers are used     //
    //    in the specified kernel                                                     //
    //    Usage:                                                                      //
    //         ./kernel_info <kernel's file name>                                     //
    //         e.g. ./kernel_info euler_qinit.cl                                      //
    //                                                                                //
    //                                                                                //
    //    Note: The compiler is doing some caching and won't compile the code again   //
    //          if it isn't necessary, which causes that the information won't print  //
    //          out sometime. A possible solution may be changing the kernel's name   //
    //   (discussed in:                                                               //
    //        https://devtalk.nvidia.com/default/topic/472539/bug-cl-nv-verbose/)     //
    // -------------------------------------------------------------------------------//
    
    cl_int            err;
    cl_platform_id    platform = GetPlatform(0); 
    cl_device_id      device = GetDevice(platform, 0);
    cl_context        context;
    cl_command_queue  commands;
    cl_program        program;

    if (argc < 2 )
    {
        std::cout<<"Kernel's name missed"<<std::endl;
        return 0;
    }

    std::string s1= "Kernel/";
    std::string s2= argv[1];
    s1.append(s2);
    std::cout<<s1<<std::endl;
    cl_char *buffer;
    size_t len;

    context = clCreateContext(0, 1, &device, NULL, NULL, &err);
    CheckError(err);
    commands = clCreateCommandQueue (context, device, 0, &err);
    CheckError(err);
    /* Create program, kernel from source */
    program = CreateProgram(LoadKernel (s1.c_str()), context);
    err     = clBuildProgram(program, 1, &device, "-cl-nv-verbose", NULL, NULL);
    
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
    buffer = new cl_char[len];
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, len, buffer, NULL);
    std::cout<<buffer<<std::endl;

    delete [] buffer;
    clReleaseProgram(program);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);
    clReleaseDevice(device);


    return 0;
}