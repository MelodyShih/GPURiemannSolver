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
    /* Opencl related variables */
    cl_int            err;
    cl_platform_id    platform = GetPlatform(0); 
    cl_device_id      device = GetDevice(platform, 0);
    int num;
    size_t size;
    clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(num), &num, NULL);
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size), &size, NULL);
    std::cout<<num<<std::endl;
    std::cout<<size<<std::endl;
    return 0;
}