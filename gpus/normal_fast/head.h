/*************************************************************************
	> File Name: opera.h
	> Author: hanlu
	> Mail: hanlu@shu.edu.cn 
	> Created Time: 2018年12月10日 星期一 21时14分15秒
 ************************************************************************/
#ifndef GUARD_opera_h
#define GUARD_opera_h

#include <iostream>
#include <cufft.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <time.h>
#include <fstream>
#include <float.h>
#include <math.h>
#include "magma_v2.h"
#include <omp.h>
#define K  5

typedef float dt;

using namespace std;

void Enc(dt *D,dt *sample,dt *PB,dt *MM,dt *MM1,int m,int n);
void Dec(dt *A,dt *B,dt *PB,int m,int n);
void printTensor(dt *A,int a,int b,int c);

void matrixProduct(dt *A,dt *B,dt *C,int a,int b,int c);
void mproduct(dt *A,dt *B,dt *C,int a,int b,int c);
__global__ void elepro(dt *t1,dt *t2,dt *tt,int k);
__global__ void elebat(dt *t1,dt *t2,dt *tt,int a,int b,int c);
__global__ void obt(dt *X,dt *sample,dt *res,int m,int r,int n);
__global__ void elepro1(dt *t1,dt *t2,dt *tt,int m,int n);
void getbatch(dt *d_A,dt *d_B,dt *d_sample,dt *d_left,dt *d_right,int m,int n,int r);
void lsq(dt *lleft,dt *rright,dt *d_res,int m,int n,int r);
void Matrixtranpose(dt *res,int gpu,int a,int b);

#endif
