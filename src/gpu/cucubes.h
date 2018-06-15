#ifndef CUCUBES_H
#define CUCUBES_H

void run_cucubes(
	int n,
	int k,
	int dimension,
	int divisions,
	int discretizations,
	int seed,
	double range,
	double pseudocount,
	double* data,
	int* decision,
	double* IGmax);

#endif
