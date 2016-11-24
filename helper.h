#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>

void CheckError (cl_int err)
{
    if (err != CL_SUCCESS) {
        std::cerr << "OpenCL call failed with error " << err << std::endl;
        exit (1);
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
    clGetPlatformInfo (id, CL_PLATFORM_NAME, 0, NULL, &size);

    std::string result;
    result.resize (size);
    clGetPlatformInfo (id, CL_PLATFORM_NAME, size,
        const_cast<char*> (result.data ()), NULL);

    return result;
}

cl_platform_id GetPlatform(int id){
    cl_uint platformIdCount = 0;
    clGetPlatformIDs (0, NULL, &platformIdCount);

    if (platformIdCount == 0) {
        std::cerr << "No OpenCL platform found" << std::endl;
    } else {
        std::cout << "Found " << platformIdCount << " platform(s)" << std::endl;
    }

    std::vector<cl_platform_id> platformIds (platformIdCount);
    clGetPlatformIDs (platformIdCount, platformIds.data (), NULL);

    /* Print out the platform information */
    for (cl_uint i = 0; i < platformIdCount; ++i) {
        std::cout << "\t (" << (i+1) << ") : " << GetPlatformName (platformIds [i]) << std::endl;
    }

    return platformIds[id];
}

std::string GetDeviceName (cl_device_id id)
{
    size_t size = 0;
    clGetDeviceInfo (id, CL_DEVICE_NAME, 0, NULL, &size);

    std::string result;
    result.resize (size);
    clGetDeviceInfo (id, CL_DEVICE_NAME, size,
        const_cast<char*> (result.data ()), NULL);

    return result;
}

cl_device_id GetDevice(cl_platform_id platform, int id){
    cl_uint deviceIdCount = 0;

    clGetDeviceIDs (platform, CL_DEVICE_TYPE_GPU, 0, NULL,
        &deviceIdCount);

    if (deviceIdCount == 0) {
        std::cerr << "No OpenCL devices found" << std::endl;
    } else {
        std::cout << "Found " << deviceIdCount << " device(s)" << std::endl;
    }

    std::vector<cl_device_id> deviceIds (deviceIdCount);
    clGetDeviceIDs (platform, CL_DEVICE_TYPE_GPU, deviceIdCount, deviceIds.data (), NULL);

    for (cl_uint i = 0; i < deviceIdCount; ++i) {
        std::cout << "\t (" << (i+1) << ") : " << GetDeviceName (deviceIds [i]) << std::endl;
    }
    return deviceIds[0];    
}

void ProgramErrMsg(cl_program program, cl_device_id device){
    size_t len;
    char buffer[4096];

    printf("Error: Failed to build program executable!\n");
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
    printf("%s\n", buffer);
    exit(1);
}

void out1(int meqn, int mbc, int mx, double xlower, double dx, double* q,
          double t, int iframe, double* aux, int maux, char* dir){
    /*output q data as the format in clawpack */
    char filename[40];
    FILE *pFile;
    sprintf(filename,"%s/_output/fort.q%04d",dir,iframe);
    pFile = fopen (filename,"w");
    fprintf(pFile, "%5d                 grid_number\n", 1);
    fprintf(pFile, "%5d                 AMR_level\n", 1);
    fprintf(pFile, "%5d                 mx\n",mx);
    fprintf(pFile, "%18.8E    xlow\n",xlower);
    fprintf(pFile, "%18.8E    dx\n\n",dx);
    for (int i = mbc; i < mx + mbc; ++i)
    {
        for (int m = 0; m < meqn; ++m)
        {
            fprintf (pFile, "%26.16E", q[meqn*i + m]);
        }
        fprintf (pFile, "\n");
    }
    fclose (pFile);

    sprintf(filename,"%s/_output/fort.t%04d",dir,iframe);
    pFile = fopen (filename,"w");
    fprintf(pFile, "%26.16E    time\n",t);
    fprintf(pFile, "%5d                 meqn\n", meqn);
    fprintf(pFile, "%5d                 ngrids\n",1);
    fprintf(pFile, "%5d                 maux\n",maux);
    fprintf(pFile, "%5d                 ndim\n",1);

    fclose (pFile);
}