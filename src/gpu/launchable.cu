#include "launchable.cuh"

void Launch(Launchable *w) {
	w->workLoop();
}

