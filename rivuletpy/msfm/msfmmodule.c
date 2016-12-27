#include <Python.h>
#include "_msfm.h"
#include "numpy/arrayobject.h"

/*This function MSFM3D calculates the shortest distance from a list of */
/*points to all other pixels in an image, using the */
/*Multistencil Fast Marching Method (MSFM). This method gives more accurate */
/*distances by using second order derivatives and cross neighbours. */
/* */
/*T=msfm3d(F, SourcePoints, UseSecond, UseCross) */
/* */
/*inputs, */
/*   F: The 3D speed image. The speed function must always be larger */
/*          than zero (min value 1e-8), otherwise some regions will
 */
/*          never be reached because the time will go to infinity.
 */
/*  SourcePoints : A list of starting points [3 x N] (distance zero) */
/*  UseSecond : Boolean Set to true if not only first but also second */
/*               order derivatives are used (default) */
/*  UseCross: Boolean Set to true if also cross neighbours */
/*               are used (default) */
/*outputs, */
/*  T : Image with distance from SourcePoints to all pixels */

/* */
/* Function is written by D.Kroon University of Twente (June 2009) */
/* Wrapped into python by Siqi Liu of Uni.Sydney (2016) */

static PyObject* msfm_run(PyObject* self, PyObject* args) {
  PyArrayObject *Fobj, *Farr, *Bobj, *Barr, *spobj, *sparr = NULL; 
  PyObject *secondobj, *crossobj = NULL;
  npy_double *F = NULL; 
  npy_double *B = NULL; 
  npy_int64 *sp = NULL;  // Pointers hold the data of numpy array
  npy_double *T, *Y = NULL;   // The pointers to the return matrices
  npy_intp *Fdims, *spdims = NULL;

  // Parse the input args
  // Expecting args: F(3D numpy array), sourcepoints (2D numpy array), second(int), cross(int)
  if (!PyArg_ParseTuple(args, "OOObb", &Fobj, &Bobj, &spobj, &secondobj, &crossobj))
    return NULL;  // TODO: raise error here

  // 1. Parse F speed image
  if (!(Farr = PyArray_FROM_OTF(Fobj, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ARRAY_ALIGNED))) return NULL;

  F = (npy_double *) PyArray_DATA(Farr);
  Fdims = PyArray_DIMS(Farr);
  int nvox = Fdims[0] * Fdims[1] * Fdims[2];

  // 2. Parse binary image
  if (!(Barr = PyArray_FROM_OTF(Bobj, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ARRAY_ALIGNED))) return NULL;

  B = (npy_int64 *) PyArray_DATA(Barr); // B and F should share the same dimensionality

  // 3. Parse source points
  if(!(sparr = PyArray_FROM_OTF(spobj, NPY_INT64, NPY_IN_ARRAY))) return NULL;

  sp = (npy_int64*) PyArray_DATA(sparr);
  spdims = PyArray_DIMS(sparr);

  // Allocate memory for outputs
  T = malloc(nvox * sizeof(npy_double));
  Y = malloc(nvox * sizeof(npy_double));

  // Convert the dims to plain int
  int Fdims_int [3] = {(int) Fdims[0], (int) Fdims[1], (int) Fdims[2]};
  int spdims_int[2];

  if (PyArray_NDIM(sparr) == 1 && spdims[0] == 3) 
  {

	spdims_int[0] = (int) spdims[0];
	spdims_int[1] = 1;
  }
  else if (PyArray_NDIM(sparr) == 2 && spdims[0] == 3)
  {
	spdims_int[0] = (int) spdims[0];
	spdims_int[1] = (int) spdims[1];
  }
  else
  {
  	PyErr_NewException("DimensionError: The dimensions of the source points matrix can only be 3X1 or 3XN", NULL, NULL);
  }


  // Run the Meaty part MSFM
  msfm3d(F, B, Fdims_int, sp, spdims_int, secondobj, crossobj, T, Y);
  PyObject* npT = PyArray_New(&PyArray_Type, 3, Fdims, NPY_DOUBLE, 0, 0, sizeof(npy_double), NPY_F_CONTIGUOUS, 0);
  memcpy(PyArray_DATA(npT), T, nvox * sizeof(double));

  // Skip the Y output for now
  // PyObject* npY =
  //     PyArray_NewFromDescr(&PyArray_Type, PyArray_DescrFromType(NPY_DOUBLE),
  //     3, Fdims, NULL,
                           // Y, NPY_FORTRAN|NPY_WRITEABLE, NULL);
  // Py_INCREF(npY);

  Py_DECREF(Farr);
  Py_DECREF(sparr);
  Py_INCREF(npT);
  free(T);
  free(Y); // Y is not used for now

  return npT;
}

static PyMethodDef msfm_methods[] = {
    {"run", (PyCFunction)msfm_run, METH_VARARGS,
     "Run multistencils fastmarching."},
    {NULL, NULL, 0, NULL}};

// Module definition
static struct PyModuleDef msfm_definition = {
    PyModuleDef_HEAD_INIT, "msfm",
    "A python module that runs multistencils fastmarching", -1, msfm_methods};

// Module initialization
PyMODINIT_FUNC PyInit_msfm(void) {
  PyObject* m;
  m = PyModule_Create(&msfm_definition);
  if (m == NULL) return NULL;

  Py_Initialize();
  import_array();
  return m;
}
