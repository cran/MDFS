#include <iostream>
#include <cstdio>
#include "cudaerrchk.cuh"
#include "allocator.cuh"

Allocator _alloc;

static std::string ptrInfo(std::string name,
	void* ptr,
	std::size_t size,
	int line,
	std::string file) {
	return name + ":" + std::to_string((long long) ptr)
		+ ", size:" + std::to_string(size)
		+ ", line:" + std::to_string(line)
		+ ", file:" + file;
}

void* Allocator::mallocHost(std::size_t size, int line, std::string file) {
	void* ptr = malloc(size);

	//std::cout << "@1@" << ptr << " " << size << " " << file << line << "\n\n";

	loc[ptr] = ptrInfo("host", ptr, size, line, file);
	return ptr;
}

void* Allocator::mallocPinned(std::size_t size, int line, std::string file) {
	void* ptr;
	CUDA(cudaMallocHost(&ptr, size));

	//std::cout << "@2@" << ptr << "\n\n";

	loc[ptr] = ptrInfo("pinned", ptr, size, line, file);
	return ptr;
}

void* Allocator::mallocDevice(std::size_t size, int line, std::string file) {
	void* ptr;
	CUDA(cudaMalloc(&ptr, size));
	loc[ptr] = ptrInfo("device", ptr, size, line, file);
	return ptr;
}

void Allocator::freeHost(void* ptr) {
	free(ptr);
	loc.erase(ptr);
}

void Allocator::freePinned(void* ptr) {
	//printf("freePinned: %lld\n", ptr);
	CUDA(cudaFreeHost(ptr));
	loc.erase(ptr);
}
void Allocator::freeDevice(void* ptr) {
	CUDA(cudaFree(ptr));
	loc.erase(ptr);
}

Allocator::~Allocator() {
	if (!loc.empty()) {
		for (auto it = loc.begin(); it != loc.end(); ++it) {
			//std::cerr << (*it).second << std::endl;
		}
	}
}
