/*
	This file is part of cpp-ethereum.

	cpp-ethereum is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	cpp-ethereum is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with cpp-ethereum.  If not, see <http://www.gnu.org/licenses/>.
*/
/** @file main.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 * Ethereum client.
 */

#include <thread>
#include <fstream>
#include <iostream>
#include "MinerAux.h"
#include "BuildInfo.h"
#include "CL/cl2.hpp"
#include <libethash/internal.h>


#include "CLMiner_kernel.h"

using namespace std;
using namespace dev;
using namespace dev::eth;
using namespace boost::algorithm;

std::vector<cl::Platform> getPlatforms()
{
	vector<cl::Platform> platforms;
	try
	{
		cl::Platform::get(&platforms);
	}
	catch(cl::Error const& err)
	{
#if defined(CL_PLATFORM_NOT_FOUND_KHR)
		if (err.err() == CL_PLATFORM_NOT_FOUND_KHR)
			cwarn << "No OpenCL platforms found";
		else
#endif
			throw err;
	}
	return platforms;
}

std::vector<cl::Device> getDevices(std::vector<cl::Platform> const& _platforms, unsigned _platformId)
{
	vector<cl::Device> devices;
	size_t platform_num = min<size_t>(_platformId, _platforms.size() - 1);
	try
	{
		_platforms[platform_num].getDevices(
			CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_ACCELERATOR,
			&devices
		);
	}
	catch (cl::Error const& err)
	{
		// if simply no devices found return empty vector
		if (err.err() != CL_DEVICE_NOT_FOUND)
			throw err;
	}
	return devices;
}

void addDefinition(string& _source, char const* _id, unsigned _value)
{
	char buf[256];
	sprintf(buf, "#define %s %uu\n", _id, _value);
	_source.insert(_source.begin(), buf, buf + strlen(buf));
}

int main(int argc, char** argv)
{
	// Set env vars controlling GPU driver behavior.
	setenv("GPU_MAX_HEAP_SIZE", "100");
	setenv("GPU_MAX_ALLOC_PERCENT", "100");
	setenv("GPU_SINGLE_ALLOC_PERCENT", "100");


	node test_node;

	h256 seed          = h256{0u};
	int epoch          = 0;
	char *kernel_data  = NULL;
	size_t kernel_size = 0;

	cl::Context m_context;
	cl::CommandQueue m_queue;

	cl::Buffer m_dag;
	cl::Buffer m_light;
	
	cl::Buffer m_header;
	cl::Buffer m_searchBuffer;
	cl::Buffer m_outputBuffer;

	vector<cl::Platform> platforms;
	vector<cl::Device> devices;
	unsigned deviceId = 0;
	char *output_string            = (char *)calloc(128, 4);

	printf("Doing OpenCL init..\n");
	{
		platforms = getPlatforms();
		if (platforms.empty())
			return -1;

		printf("We have a total of %ld platforms...\n", platforms.size());

		// get GPU device of the default platform
		devices  = getDevices(platforms, 0);
		if (devices.empty())
		{
			std::cout << "No OpenCL devices found.";
			return -1;
		}

		printf("We have a total of %ld devices, using device %ld...\n", devices.size(), deviceId);
	}

	cl::Device& device = devices[min<unsigned>(deviceId, devices.size() - 1)];
	string device_version = device.getInfo<CL_DEVICE_VERSION>();
	std::cout << "Device:   " << device.getInfo<CL_DEVICE_NAME>() << " / " << device_version << "\n";

	// Create context and queue
	m_context = cl::Context(vector<cl::Device>(&device, &device + 1));
	m_queue   = cl::CommandQueue(m_context, device);
	

	printf("Setting epoch: %d\n", epoch);
	for(int i = 0; i < epoch; i++) seed = sha3(seed);

	EthashAux::LightType light = EthashAux::light(seed);
	printf("generating dag for block block_number %ld \n", light->light->block_number);

	uint64_t dagSize = ethash_get_datasize(light->light->block_number);
	uint32_t dagSize128 = (unsigned)(dagSize / ETHASH_MIX_BYTES);
	uint32_t lightSize64 = (unsigned)(light->data().size() / sizeof(node));

	printf("Dagsize is: %ld MB \n", dagSize/1024/1024);

	string code(CLMiner_kernel, CLMiner_kernel + sizeof(CLMiner_kernel));
	addDefinition(code, "GROUP_SIZE", 64);
	addDefinition(code, "DAG_SIZE", dagSize128);
	addDefinition(code, "LIGHT_SIZE", lightSize64);
	addDefinition(code, "ACCESSES", ETHASH_ACCESSES);
	addDefinition(code, "MAX_OUTPUTS", 1);
	// addDefinition(code, "PLATFORM", platformId);
	// addDefinition(code, "COMPUTE", computeCapability);
	// addDefinition(code, "THREADS_PER_HASH", s_threadsPerHash);


	m_header = cl::Buffer(m_context, CL_MEM_READ_ONLY, 32);
	m_searchBuffer = cl::Buffer(m_context, CL_MEM_WRITE_ONLY, (64) * sizeof(uint32_t));
	m_outputBuffer = cl::Buffer(m_context, CL_MEM_WRITE_ONLY, (1024) * sizeof(uint32_t));

	std::cout << "Creating light cache buffer, size: " << light->data().size() << "\n";
	try {
		m_light = cl::Buffer(m_context, CL_MEM_READ_ONLY, light->data().size());
	} catch(...) {
		std::cout << "Couldn't alloc light data? \n";
		return -1;
	}

	std::cout << "Creating DAG buffer, size: " << dagSize << "\n";
	try {
	m_dag = cl::Buffer(m_context, CL_MEM_READ_ONLY, dagSize);

	}catch(...){
		std::cout << "Couldn't alloc dag data?\n";
		return -1;
	}

	
	ethash_calculate_dag_item(&test_node, 0x23599F80>>6, light->light);
	for(int i = 0; i < 64/4; i++) {
		printf("0x%08X\n", test_node.words[i]);
	}

	uint32_t	test_header[32/4] = {
		0xAAAAAAA0, 0x0BBBBBB0,
		0xDEADBEEF, 0xBEEF4DAD,
		0xBEEFBEEF, 0xACA74DAD,
		0xDAD5CAFE, 0xDAD5B00B
	} ;

	m_queue.enqueueWriteBuffer(m_header, CL_TRUE, 0, 32, test_header);
	m_queue.enqueueWriteBuffer(m_light, CL_TRUE, 0, light->data().size(), light->data().data());
	

	std::cout << "Wrote light data, loading kernel...\n";

	cl::Program::Sources sources{{code.data(), code.size()}};
	cl::Program program(m_context, sources);
	cl::Kernel m_dagKernel;
	try
	{
		program.build({device}, "");
		std::cout << "Build info:" << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << "\n";
	}
	catch (cl::Error const&)
	{
		std::cout << "Build error:" << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << "\n";
		return -1;
	}

	m_dagKernel = cl::Kernel(program, "ethash_calculate_dag_item");

	std::cout << "got kernel, generating dag...\n";

	unsigned m_workgroupSize = 64;
	unsigned m_globalWorkSize = 8192 * m_workgroupSize;

	uint32_t const work = (uint32_t)(dagSize / sizeof(node));
	uint32_t fullRuns = work / m_globalWorkSize;
	uint32_t const restWork = work % m_globalWorkSize;
	if (restWork > 0) fullRuns++;

	m_dagKernel.setArg(1, m_light);
	m_dagKernel.setArg(2, m_dag);
	m_dagKernel.setArg(3, ~0u);

	for (uint32_t i = 0; i < fullRuns; i++)
	{
	 	m_dagKernel.setArg(0, i * m_globalWorkSize);
	 	m_queue.enqueueNDRangeKernel(m_dagKernel, cl::NullRange, m_globalWorkSize, m_workgroupSize);
	 	m_queue.finish();
	 	std::cout << "DAG" << int(100.0f * i / fullRuns) << "%" << "\n";
	}


	// Finally, enter the test loop
	do {
	    printf("Testing n' testing kernel...");
	
		// Kernel load scope
		{
			FILE *fp = fopen("test.bin", "rb");
			if(fp == NULL) {
				printf("Couldn't load binary kernel... le epic dying xD\n");
				continue;
			}

			fseek(fp, 0L, SEEK_END);
			kernel_size = ftell(fp);
			fseek(fp, 0L, SEEK_SET);

			kernel_data = (char *)calloc(kernel_size, 1);
			size_t bytes = fread(kernel_data, 1, kernel_size, fp);
			fclose(fp);
			if(bytes != kernel_size) {
				printf("Couldn't read entire file (%zu, %zu)\n", kernel_size, bytes);
				free(kernel_data);
				continue;
			}

			printf("Kernel size: %zu\n", kernel_size);
		}


		std::vector<unsigned char> bin_data(kernel_data, kernel_data + kernel_size);
		cl::Program::Binaries binaries{{ bin_data } };
		cl::Program asmProgram; 
		cl::Kernel m_asmKernel;

		printf("Loading binary kernel test.bin\n");
		{
			cl::Program program(m_context, { device }, binaries);
			try
			{
				program.build({ device }, "");
				asmProgram = program;
				std::cout << "Build info:" << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << "\n";
			}
			catch (cl::Error const&)
			{
				std::cout << "Build info:" << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << "\n";
				free(kernel_data);
				continue;
			}
		}
		free(kernel_data);

		printf("Setting up params for run...\n");
		try
		{
			m_asmKernel = cl::Kernel(asmProgram, "keccak_search");
		 	m_asmKernel.setArg(0, m_outputBuffer);
		 	m_asmKernel.setArg(1, m_header);
		 	m_asmKernel.setArg(2, m_dag);

			cl_ulong target = 0x80000000090000L; 
			cl_ulong nonce = 0x133700001338000L;
			cl_uint isolate = 666;
			cl_uint factor  = (1UL << 32)/dagSize128;
		 	m_asmKernel.setArg(3, nonce);
		 	m_asmKernel.setArg(4, target);
		 	m_asmKernel.setArg(5, isolate);
		 	m_asmKernel.setArg(6, dagSize128);
		 	m_asmKernel.setArg(7, factor);



		}catch (cl::Error const&) {
			printf("Failed to load and set args for kernel...\n");
		}

		printf("Successful load, running...\n");

		try{
		 	m_queue.enqueueNDRangeKernel(m_asmKernel, cl::NullRange, 128, 64);
		 	m_queue.finish();
			// get data
		 }

		catch (cl::Error const& err)
		{
			std::cout << err.what() << "(" << err.err() << ")";
			continue;
		}
		printf("Successful run, reading...\n");

		try {
			m_queue.enqueueReadBuffer(m_outputBuffer, CL_TRUE, 0, 128 * sizeof(uint), output_string);
		}catch(cl::Error const& err){
			std::cout << err.what() << "(" << err.err() << ")";
			continue;
		}

		for(int i = 0; i < 8*2; i++) {
			for(int j = 0; j < 8; j++) {
				printf("%08X-", ((uint*)output_string)[i*8 + j]);
			}
			printf("\n");
		}

	}while(getchar() != 'q');
	free(output_string);
	return 0;
}
