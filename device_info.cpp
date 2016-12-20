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
    size_t size1;
    clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(num), &num, NULL);
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size), &size, NULL);
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size1), &size1, NULL);
    
    std::cout<<"CL_DEVICE_MAX_COMPUTE_UNITS = "<<num<<std::endl;
    std::cout<<"CL_DEVICE_MAX_WORK_GROUP_SIZE = "<<size<<std::endl;
    std::cout<<"CL_DEVICE_MAX_WORK_ITEM_SIZES = "<<size1<<std::endl;
    return 0;
}