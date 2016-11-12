#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>

void CheckError (cl_int err)
{
	if (err != CL_SUCCESS) {
		std::cerr << "OpenCL call failed with error " << err << std::endl;
		std::exit (1);
	}
}

std::string LoadKernel (const char* name)
{
	std::ifstream in(name);
	std::string result ((std::istreambuf_iterator<char> (in)),std::istreambuf_iterator<char> ());
	
	return result;
}

cl_program CreateProgram (const std::string& source, cl_context context)
{
	const size_t lengths[1] = { source.size () };
	const char* sources[1] = { source.data () };

	cl_int err = 0;
	cl_program program = clCreateProgramWithSource (context, 1, sources, lengths, &err);

	CheckError (err);

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

cl_platform_id GetPlatform(int id){
	cl_uint platformIdCount = 0;
	clGetPlatformIDs (0, nullptr, &platformIdCount);

	if (platformIdCount == 0) {
		std::cerr << "No OpenCL platform found" << std::endl;
	} else {
		std::cout << "Found " << platformIdCount << " platform(s)" << std::endl;
	}

	std::vector<cl_platform_id> platformIds (platformIdCount);
	clGetPlatformIDs (platformIdCount, platformIds.data (), nullptr);

	/* Print out the platform information */
	for (cl_uint i = 0; i < platformIdCount; ++i) {
		std::cout << "\t (" << (i+1) << ") : " << GetPlatformName (platformIds [i]) << std::endl;
	}

	return platformIds[id];
}

cl_device_id GetDevice(cl_platform_id platform, int id){
	cl_uint deviceIdCount = 0;

	clGetDeviceIDs (platform, CL_DEVICE_TYPE_GPU, 0, nullptr,
		&deviceIdCount);

	if (deviceIdCount == 0) {
		std::cerr << "No OpenCL devices found" << std::endl;
	} else {
		std::cout << "Found " << deviceIdCount << " device(s)" << std::endl;
	}

	std::vector<cl_device_id> deviceIds (deviceIdCount);
	clGetDeviceIDs (platform, CL_DEVICE_TYPE_GPU, deviceIdCount, deviceIds.data (), nullptr);

	for (cl_uint i = 0; i < deviceIdCount; ++i) {
		std::cout << "\t (" << (i+1) << ") : " << GetDeviceName (deviceIds [i]) << std::endl;
	}
	return deviceIds[0];	
}

void ProgramErrMsg(cl_program program, cl_device_id device){
    size_t len;
    char buffer[2048];

    printf("Error: Failed to build program executable!\n");
	clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
    printf("%s\n", buffer);
    exit(1);
}