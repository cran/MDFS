#ifndef CALC_CUH
#define CALC_CUH

#include <mutex>
#include <vector>
#include <thread>
#include "worker.cuh"
#include "discretizer.cuh"
#include "scheduler.cuh"
#include "launchconfig.cuh"

class Calc {
	std::mutex mutex;
	LaunchConfig lc;
	InputFile inf;
	double* IGmax;
	void (*pC)(int, int, int, int);
	Scheduler scheduler;
	Discretizer discretizer;

	DataFile df;
	std::vector<Worker*> workerlist;
	std::vector<std::thread> threads;

public:
	Calc(LaunchConfig lc,
		InputFile inf,
		double* IGmax,
		void (*progressCallback)(int, int, int, int));

	void returnResults(float* IG);
};

#endif
