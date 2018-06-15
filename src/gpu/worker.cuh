#ifndef WORKER_CUH
#define WORKER_CUH

#include <stdint.h>
#include <vector>
#include "launchable.cuh"
#include "launchconfig.cuh"
#include "scheduler.cuh"

class Calc;

class Worker : public Launchable {
	const int workerId;
	int vars;
	int gpuId;
	LaunchConfig lc;
	Calc* calc;
	Scheduler* scheduler;
	DataFile df;

	uint64_t** data[2];
	uint64_t** devData[2];

	uint64_t** devCounters[2];

	float* IG[2];
	float* devIG[2];
	cudaStream_t stream[2];

	std::vector<int> decodePackId(uint64_t packId);
	void lowerDimCounters(int ix, std::vector<int> offset, int strNum);

	void process(uint64_t packId, int strNum);
public:
	Worker(int vars,
		int gpuId,
		LaunchConfig lc,
		Calc* calc,
		Scheduler* scheduler,
		DataFile df);
	~Worker();

	void workLoop();
};

#endif
