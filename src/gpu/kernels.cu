#include "kernels.cuh"
#include "kernels2D.cuh"
#include "kernels3D.cuh"
#include "kernels4D.cuh"
#include "kernels5D.cuh"
#include "tableskernel.cuh"
#include "splitkernel.cuh"

std::vector<std::pair<KernelParam, void(*)(KernelParam, cudaStream_t)>> kernels;

std::ostream& operator<< (std::ostream& out, KernelParam const& prop) {
	out << "tileSize:" << prop.tileSize << " ";
	out << "dim:" << prop.dim << " ";
	out << "div:" << prop.div << " ";
	out << "rm:" << prop.rm << " ";
	out << "bf:" << prop.bf << " ";
	out << "tablesKernel:" << prop.tablesKernel << " ";
	out << (prop.objs[0] < (1 << 16) ? "objc0 < 2^16 " : "objc0 >= 2^16 ");
	out << (prop.objs[1] < (1 << 16) ? "objc1 < 2^16 " : "objc1 >= 2^16 ");
	return out;
}

bool operator==(const KernelParam& lhs, const KernelParam& rhs) {
	bool lobjs = (lhs.objs[0] < (1 << 16)) && (lhs.objs[1] < (1 << 16));
	bool robjs = (rhs.objs[0] < (1 << 16)) && (rhs.objs[1] < (1 << 16));

	return (lobjs == robjs) &&
		(lhs.tileSize == rhs.tileSize) &&
		(lhs.dim == rhs.dim) &&
		(lhs.div == rhs.div) &&
		(lhs.rm == rhs.rm) &&
		(lhs.bf == rhs.bf) &&
		(lhs.tablesKernel == rhs.tablesKernel);
}

KernelParam::KernelParam(LaunchConfig lc, bool tablesKernel, int index,
	int vars, uint64_t** data, uint64_t** counters,
	std::vector<int> offset, int packs0, int packs1,
	int objs0, int objs1, float* IG)
	: tileSize(lc.tileSize), dim(lc.dim), div(lc.div),
	rm(lc.rm), bf(lc.bf), tablesKernel(tablesKernel),
	index(index), disc(lc.disc), vars(vars), IG(IG) {

	objs[0] = objs0;
	objs[1] = objs1;

	for (int i = 0; i < dim; i++) {
		this->data[i] = data[i];
		this->counters[i] = counters[i];
		this->offset[i] = offset[i];
	}

	packs[0] = packs0;
	packs[1] = packs1;

	const float objsmin = std::min(objs0, objs1);

	for (int i = 0; i < 2; i++) {
		pseudo[i] = ((float)objs[i] / objsmin) * lc.pseudo;
	}
}

KernelParam::KernelParam(int tileSize, int dim, int div, ReduceMethod rm,
	BinaryFormat bf, bool tablesKernel, int objs0)
	: tileSize(tileSize), dim(dim), div(div), rm(rm), bf(bf),
	tablesKernel(tablesKernel)  {

	for (int i = 0; i < 2; i++) {
		objs[i] = objs0;
	}
}

bool init() {
#define KERNELS2D(DIV) \
	kernels.push_back(std::make_pair(KernelParam(512, 2, (DIV), RM_AVG, BF_SHIFT, false,     0), kernel2DWrapper<512, (DIV), 1, 1>));\
	kernels.push_back(std::make_pair(KernelParam(512, 2, (DIV), RM_MAX, BF_SHIFT, false,     0), kernel2DWrapper<512, (DIV), 0, 1>));\
	kernels.push_back(std::make_pair(KernelParam(512, 2, (DIV), RM_AVG, BF_SHIFT, false, 1<<16), kernel2DWrapper<512, (DIV), 1, 0>));\
	kernels.push_back(std::make_pair(KernelParam(512, 2, (DIV), RM_MAX, BF_SHIFT, false, 1<<16), kernel2DWrapper<512, (DIV), 0, 0>));
#define KERNELS3D(DIV) \
	kernels.push_back(std::make_pair(KernelParam(64, 3, (DIV), RM_AVG, BF_SHIFT, false,     0), kernel3DWrapper<64, (DIV), 1, 1>));\
	kernels.push_back(std::make_pair(KernelParam(64, 3, (DIV), RM_MAX, BF_SHIFT, false,     0), kernel3DWrapper<64, (DIV), 0, 1>));\
	kernels.push_back(std::make_pair(KernelParam(64, 3, (DIV), RM_AVG, BF_SHIFT, false, 1<<16), kernel3DWrapper<64, (DIV), 1, 0>));\
	kernels.push_back(std::make_pair(KernelParam(64, 3, (DIV), RM_MAX, BF_SHIFT, false, 1<<16), kernel3DWrapper<64, (DIV), 0, 0>));

#define SPLITKERNELS(TS, DIM, DIV, BITS) \
	kernels.push_back(std::make_pair(KernelParam((TS),(DIM),(DIV),RM_AVG,BF_SPLIT,false,    0),splitKernelWrapper<(TS),(DIM),(DIV),(BITS),1,1>));\
	kernels.push_back(std::make_pair(KernelParam((TS),(DIM),(DIV),RM_MAX,BF_SPLIT,false,    0),splitKernelWrapper<(TS),(DIM),(DIV),(BITS),0,1>));\
	kernels.push_back(std::make_pair(KernelParam((TS),(DIM),(DIV),RM_AVG,BF_SPLIT,false,1<<16),splitKernelWrapper<(TS),(DIM),(DIV),(BITS),1,0>));\
	kernels.push_back(std::make_pair(KernelParam((TS),(DIM),(DIV),RM_MAX,BF_SPLIT,false,1<<16),splitKernelWrapper<(TS),(DIM),(DIV),(BITS),0,0>));

	KERNELS2D(10)
	KERNELS2D(11)
	KERNELS2D(12)
	KERNELS2D(13)
	KERNELS2D(14)
	KERNELS2D(15)

	KERNELS3D(5)

	SPLITKERNELS(512, 2, 1, 1)
	SPLITKERNELS(512, 2, 2, 2)
	SPLITKERNELS(512, 2, 3, 2)
	SPLITKERNELS(512, 2, 4, 4)
	SPLITKERNELS(512, 2, 5, 4)
	SPLITKERNELS(512, 2, 6, 4)
	SPLITKERNELS(512, 2, 7, 4)
	SPLITKERNELS(512, 2, 8, 4)
	SPLITKERNELS(512, 2, 9, 4)

	SPLITKERNELS(64, 3, 1, 1)
	SPLITKERNELS(64, 3, 2, 2)
	SPLITKERNELS(64, 3, 3, 2)
	SPLITKERNELS(64, 3, 4, 4)

	SPLITKERNELS(32, 4, 1, 1)
	SPLITKERNELS(32, 4, 2, 2)
	SPLITKERNELS(32, 4, 3, 2)

	SPLITKERNELS(8, 5, 1, 1)
	SPLITKERNELS(8, 5, 2, 2)

	// FIXME?
	kernels.push_back(std::make_pair(KernelParam(64, 3, 1, RM_AVG, BF_SPLIT, true,     0), tablesKernelWrapper<64, 2, 1, 1, 1>));
	kernels.push_back(std::make_pair(KernelParam(64, 3, 1, RM_MAX, BF_SPLIT, true,     0), tablesKernelWrapper<64, 2, 1, 1, 1>));
	kernels.push_back(std::make_pair(KernelParam(64, 3, 1, RM_AVG, BF_SPLIT, true, 1<<16), tablesKernelWrapper<64, 2, 1, 1, 0>));
	kernels.push_back(std::make_pair(KernelParam(64, 3, 1, RM_MAX, BF_SPLIT, true, 1<<16), tablesKernelWrapper<64, 2, 1, 1, 0>));

	kernels.push_back(std::make_pair(KernelParam(64, 3, 2, RM_AVG, BF_SPLIT, true,     0), tablesKernelWrapper<64, 2, 2, 2, 1>));
	kernels.push_back(std::make_pair(KernelParam(64, 3, 2, RM_MAX, BF_SPLIT, true,     0), tablesKernelWrapper<64, 2, 2, 2, 1>));
	kernels.push_back(std::make_pair(KernelParam(64, 3, 2, RM_AVG, BF_SPLIT, true, 1<<16), tablesKernelWrapper<64, 2, 2, 2, 0>));
	kernels.push_back(std::make_pair(KernelParam(64, 3, 2, RM_MAX, BF_SPLIT, true, 1<<16), tablesKernelWrapper<64, 2, 2, 2, 0>));

	kernels.push_back(std::make_pair(KernelParam(64, 3, 3, RM_AVG, BF_SPLIT, true,     0), tablesKernelWrapper<64, 2, 3, 2, 1>));
	kernels.push_back(std::make_pair(KernelParam(64, 3, 3, RM_MAX, BF_SPLIT, true,     0), tablesKernelWrapper<64, 2, 3, 2, 1>));
	kernels.push_back(std::make_pair(KernelParam(64, 3, 3, RM_AVG, BF_SPLIT, true, 1<<16), tablesKernelWrapper<64, 2, 3, 2, 0>));
	kernels.push_back(std::make_pair(KernelParam(64, 3, 3, RM_MAX, BF_SPLIT, true, 1<<16), tablesKernelWrapper<64, 2, 3, 2, 0>));

	kernels.push_back(std::make_pair(KernelParam(64, 3, 4, RM_AVG, BF_SPLIT, true,     0), tablesKernelWrapper<64, 2, 4, 4, 1>));
	kernels.push_back(std::make_pair(KernelParam(64, 3, 4, RM_MAX, BF_SPLIT, true,     0), tablesKernelWrapper<64, 2, 4, 4, 1>));
	kernels.push_back(std::make_pair(KernelParam(64, 3, 4, RM_AVG, BF_SPLIT, true, 1<<16), tablesKernelWrapper<64, 2, 4, 4, 0>));
	kernels.push_back(std::make_pair(KernelParam(64, 3, 4, RM_MAX, BF_SPLIT, true, 1<<16), tablesKernelWrapper<64, 2, 4, 4, 0>));

	kernels.push_back(std::make_pair(KernelParam(32, 4, 1, RM_AVG, BF_SPLIT, true,     0), tablesKernelWrapper<32, 3, 1, 1, 1>));
	kernels.push_back(std::make_pair(KernelParam(32, 4, 1, RM_MAX, BF_SPLIT, true,     0), tablesKernelWrapper<32, 3, 1, 1, 1>));
	kernels.push_back(std::make_pair(KernelParam(32, 4, 1, RM_AVG, BF_SPLIT, true, 1<<16), tablesKernelWrapper<32, 3, 1, 1, 0>));
	kernels.push_back(std::make_pair(KernelParam(32, 4, 1, RM_MAX, BF_SPLIT, true, 1<<16), tablesKernelWrapper<32, 3, 1, 1, 0>));

	kernels.push_back(std::make_pair(KernelParam(32, 4, 2, RM_AVG, BF_SPLIT, true,     0), tablesKernelWrapper<32, 3, 2, 2, 1>));
	kernels.push_back(std::make_pair(KernelParam(32, 4, 2, RM_MAX, BF_SPLIT, true,     0), tablesKernelWrapper<32, 3, 2, 2, 1>));
	kernels.push_back(std::make_pair(KernelParam(32, 4, 2, RM_AVG, BF_SPLIT, true, 1<<16), tablesKernelWrapper<32, 3, 2, 2, 0>));
	kernels.push_back(std::make_pair(KernelParam(32, 4, 2, RM_MAX, BF_SPLIT, true, 1<<16), tablesKernelWrapper<32, 3, 2, 2, 0>));

	kernels.push_back(std::make_pair(KernelParam(32, 4, 3, RM_AVG, BF_SPLIT, true,     0), tablesKernelWrapper<32, 3, 3, 2, 1>));
	kernels.push_back(std::make_pair(KernelParam(32, 4, 3, RM_MAX, BF_SPLIT, true,     0), tablesKernelWrapper<32, 3, 3, 2, 1>));
	kernels.push_back(std::make_pair(KernelParam(32, 4, 3, RM_AVG, BF_SPLIT, true, 1<<16), tablesKernelWrapper<32, 3, 3, 2, 0>));
	kernels.push_back(std::make_pair(KernelParam(32, 4, 3, RM_MAX, BF_SPLIT, true, 1<<16), tablesKernelWrapper<32, 3, 3, 2, 0>));

	kernels.push_back(std::make_pair(KernelParam(8, 5, 1, RM_AVG, BF_SPLIT, true,     0), tablesKernelWrapper<8, 4, 1, 1, 1>));
	kernels.push_back(std::make_pair(KernelParam(8, 5, 1, RM_MAX, BF_SPLIT, true,     0), tablesKernelWrapper<8, 4, 1, 1, 1>));
	kernels.push_back(std::make_pair(KernelParam(8, 5, 1, RM_AVG, BF_SPLIT, true, 1<<16), tablesKernelWrapper<8, 4, 1, 1, 0>));
	kernels.push_back(std::make_pair(KernelParam(8, 5, 1, RM_MAX, BF_SPLIT, true, 1<<16), tablesKernelWrapper<8, 4, 1, 1, 0>));

	kernels.push_back(std::make_pair(KernelParam(8, 5, 2, RM_AVG, BF_SPLIT, true,     0), tablesKernelWrapper<8, 4, 2, 2, 1>));
	kernels.push_back(std::make_pair(KernelParam(8, 5, 2, RM_MAX, BF_SPLIT, true,     0), tablesKernelWrapper<8, 4, 2, 2, 1>));
	kernels.push_back(std::make_pair(KernelParam(8, 5, 2, RM_AVG, BF_SPLIT, true, 1<<16), tablesKernelWrapper<8, 4, 2, 2, 0>));
	kernels.push_back(std::make_pair(KernelParam(8, 5, 2, RM_MAX, BF_SPLIT, true, 1<<16), tablesKernelWrapper<8, 4, 2, 2, 0>));

	return true;
}

static const bool doneInit = init();

