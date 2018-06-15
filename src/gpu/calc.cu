#include "calc.cuh"
#include "launchable.cuh"
#include "cudaerrchk.cuh"

Calc::Calc(LaunchConfig lc,
	InputFile inf,
	double* IGmax,
	void (*progressCallback)(int, int, int, int)) :
	lc(lc), inf(inf), IGmax(IGmax), pC(progressCallback),
	scheduler(lc.dim, (inf.vars + lc.tileSize - 1) / lc.tileSize, pC),
	discretizer(lc, inf, &scheduler) {

	df = discretizer.getDataFile();

	memset(IGmax, 0, inf.vars * sizeof(double));

	int workerCount;
	CUDA(cudaGetDeviceCount(&workerCount));

	for (int i = 0; i < workerCount; ++i) {
		workerlist.push_back(new Worker(inf.vars, i, lc, this, &scheduler, df));
	}

	if (workerlist.size() == 0) {
		return;
	}

	threads.push_back(std::thread(Launch, &discretizer));

	for (auto i : this->workerlist) {
		threads.push_back(std::thread(Launch, i));
	}
	for (auto& t : this->threads) {
		t.join();
	}

	for (auto i : this->workerlist) {
		delete i;
	}
}

void Calc::returnResults(float* IG) {
	mutex.lock();
	for (int i = 0; i < inf.vars; i++) {
		IGmax[i] = std::max(IGmax[i], (double) IG[i]);
	}
	mutex.unlock();
}

