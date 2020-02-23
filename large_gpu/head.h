/*************************************************************************
	> File Name: opera.h
	> Author: hanlu
	> Mail: hanlu@shu.edu.cn 
	> Created Time: 2018年12月10日 星期一 21时14分15秒
 ************************************************************************/
#ifndef GUARD_opera_h
#define GUARD_opera_h

#include <iostream>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <time.h>
#include <fstream>
#include <float.h>
#include <math.h>
#include "magma_v2.h"
#define K  5
typedef float dt;

using namespace std;

void printTensor(dt *A,int a,int b,int c);
void signalpro(dt *A,dt *C,int a,int b);
void signalpro1(dt *A,dt *B,dt *C,int a,int b);
void newmode(dt *A,dt *B,dt *sample,dt *left,dt *right,int m,int n,int r);

void Enc(dt *D,dt *sample,dt *PB,dt *MM,dt *MM1,int m,int n);
void Dec(dt *A,dt *B,dt *PB,int m,int n);
void matrixProduct(dt *A,dt *B,dt *C,int a,int b,int c);
void mproduct(dt *A,dt *B,dt *C,int a,int b,int c);
__global__ void elepro(dt *t1,dt *t2,dt *tt,int k);
__global__ void elelbat(dt *t1,dt *t2,dt *tt,int a,int b,int c);
void getbatch(dt *A,dt *B,dt *sample,dt *left,dt *right,int m,int n,int r);
void lsq(dt *left,dt *right,dt *res,int m,int n,int r);
void Matrixtranpose(dt *res,int a,int b);

#endif
