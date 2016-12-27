#include "Python.h"
#include "numpy/arrayobject.h"
#include <stdbool.h>

#ifndef _MSFM
#define _MSFM
#endif

void msfm3d(npy_double* F,             // The input speed image
	        npy_int64* bimg,
            int dims[3],           // The size of the input speed image
            npy_int64* SourcePoints,  // The source points
            int dims_sp[3],        // The size of the source point array
            bool usesecond, bool usecross,
            npy_double* T,  // The output time crosing map
            npy_double* Y); // The output euclidean image
