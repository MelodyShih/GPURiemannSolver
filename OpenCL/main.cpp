#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>

#include "OpenCL/opencl.h"

#define DATA_SIZE (16)

void CheckError (cl_int err, std::string message)
{
	if (err != CL_SUCCESS) {
		std::cerr << "OpenCL call failed with error " << err << std::endl;
		std::cerr << message << std::endl;
		std::exit (1);
	}
}

std::string LoadKernel (const char* name)
{
	std::ifstream in(name);
	std::string result ((std::istreambuf_iterator<char> (in)),std::istreambuf_iterator<char> ());
	
	// std::cout<<result<<std::endl;
	
	return result;
}

cl_program CreateProgram (const std::string& source, cl_context context)
{
	const size_t lengths[1] = { source.size () };
	const char* sources[1] = { source.data () };

	cl_int err = 0;
	cl_program program = clCreateProgramWithSource (context, 1, sources, lengths, &err);

	CheckError (err, "fail to create program");

	return program;
}

std::string GetPlatformName (cl_platform_id id)
{
	size_t size = 0;
	clGetPlatformInfo (id, CL_PLATFORM_NAME, 0, nullptr, &size);

	std::string result;
	result.resize (size);
	clGetPlatformInfo (id, CL_PLATFORM_NAME, size,
		const_cast<char*> (result.data ()), nullptr);

	return result;
}

std::string GetDeviceName (cl_device_id id)
{
	size_t size = 0;
	clGetDeviceInfo (id, CL_DEVICE_NAME, 0, nullptr, &size);

	std::string result;
	result.resize (size);
	clGetDeviceInfo (id, CL_DEVICE_NAME, size,
		const_cast<char*> (result.data ()), nullptr);

	return result;
}

int main(int argc, char const *argv[])
{
/* Get Platform */
	cl_uint platformIdCount = 0;
	clGetPlatformIDs (0, nullptr, &platformIdCount);

	if (platformIdCount == 0) {
		std::cerr << "No OpenCL platform found" << std::endl;
		return 1;
	} else {
		std::cout << "Found " << platformIdCount << " platform(s)" << std::endl;
	}

	std::vector<cl_platform_id> platformIds (platformIdCount);
	clGetPlatformIDs (platformIdCount, platformIds.data (), nullptr);

	for (cl_uint i = 0; i < platformIdCount; ++i) {
		std::cout << "\t (" << (i+1) << ") : " << GetPlatformName (platformIds [i]) << std::endl;
	}
/* Get Device */
	cl_uint deviceIdCount = 0;
	clGetDeviceIDs (platformIds [0], CL_DEVICE_TYPE_ALL, 0, nullptr,
		&deviceIdCount);

	if (deviceIdCount == 0) {
		std::cerr << "No OpenCL devices found" << std::endl;
		return 1;
	} else {
		std::cout << "Found " << deviceIdCount << " device(s)" << std::endl;
	}

	std::vector<cl_device_id> deviceIds (deviceIdCount);
	clGetDeviceIDs (platformIds [0], CL_DEVICE_TYPE_ALL, deviceIdCount,
		deviceIds.data (), nullptr);

	for (cl_uint i = 0; i < deviceIdCount; ++i) {
		std::cout << "\t (" << (i+1) << ") : " << GetDeviceName (deviceIds [i]) << std::endl;
	}

/* Create context */
	const cl_context_properties contextProperties [] =
	{
		CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties> (platformIds [0]),
		0, 0
	};

	cl_int err = CL_SUCCESS;
	cl_context context = clCreateContext (contextProperties, deviceIdCount,
		                                  deviceIds.data (), nullptr, nullptr, &err);
	CheckError (err, "fail to create context");


/* --------------------------------------------------------------------------------------------*/
//																							   //
//                          I can start programming here                                       //
//																							   //
/* --------------------------------------------------------------------------------------------*/

/* Create program from source */
	cl_program program = CreateProgram(LoadKernel ("Kernel/qinit.cl"), context);

	err = clBuildProgram(program, deviceIdCount, deviceIds.data(), NULL, NULL, NULL);
	if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];

        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(program, deviceIds[0], CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        exit(1);
    }

/* Create Kernel */ 
	cl_kernel kernel = clCreateKernel(program, "square", &err);
	CheckError(err,"fail to create kernel");

/* Create data and run the kernel */
    int meqn = 2;
	int mx = DATA_SIZE, mbc = 2;
	int count = mx + 2*mbc;
	double p[count], u[count];
	double results[count];

    std::size_t global = count;
    std::size_t local;

/* Initialize data */
	for(int i = 0; i < count; i++){
        if(i == 2){
        	p[i] = 2;
        	u[i] = 0;
		}
		else{
			p[i] = 0;
			u[i] = 0;
		}
	}

/* Allocate device memory */
	cl_mem d_p, d_u;                 // device memory used for the input array
    cl_mem d_o;                      // device memory used for the output array

	d_p  = clCreateBuffer(context, CL_MEM_READ_ONLY,  sizeof(double)*count, NULL, NULL);
	d_u  = clCreateBuffer(context, CL_MEM_READ_ONLY,  sizeof(double)*count, NULL, NULL);
    d_o  = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(double)*count, NULL, NULL);

    if (!d_p || !d_o || !d_u)
    {
        printf("Error: Failed to allocate device memory!\n");
        exit(1);
    }

/* Create commands */
    cl_command_queue commands = clCreateCommandQueue (context, deviceIds[0], 0, &err);
    CheckError(err, "");

/* Write data set into the input array in device memory */
    CheckError(clEnqueueWriteBuffer(commands, d_p, CL_TRUE, 0, sizeof(double)*count, p, 0, NULL, NULL), "write data");
    CheckError(clEnqueueWriteBuffer(commands, d_u, CL_TRUE, 0, sizeof(double)*count, u, 0, NULL, NULL), "write data");

/* Set kernel arguments */
    CheckError(clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_p), "set kernel arg");
    CheckError(clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_u), "set kernel arg");
    CheckError(clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_o), "set kernel arg");
    CheckError(clSetKernelArg(kernel, 3, sizeof(int)   , &count), "set kernel arg");

/* Retrieve kernel work group info */
    CheckError(clGetKernelWorkGroupInfo(kernel, deviceIds[0], CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL), 
    	       "Retrieve work group info");
    std::cout<<local<<std::endl;

/* Launch kernel */
	CheckError(clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, NULL, 0, NULL, NULL),"");
	//CheckError(clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, &local, 0, NULL, NULL),"");

/* Read ouput array */
    CheckError(clEnqueueReadBuffer(commands, d_o, CL_TRUE, 0, sizeof(double)*count, results, 0, NULL, NULL ),"");  

/* Print out result */
    for (int i = 0; i < count; ++i)
    {
    	std::cout<<"q["<<i<<"]"<<" = "<<results[i]<<std::endl;
    }

    // ofstream myfile;
    // myfile.open ("fort.q0000");
    // myfile << "Writing this to a file.\n";


    clReleaseMemObject(d_u);
    clReleaseMemObject(d_p);
    clReleaseMemObject(d_o);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);

	return 0;
}