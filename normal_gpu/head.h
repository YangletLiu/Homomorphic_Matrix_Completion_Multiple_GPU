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

typedef float dt;

using namespace std;

void printTensor(dt *A,int a,int b,int c);

void matrixProduct(dt *A,dt *B,dt *C,int a,int b,int c);
void mproduct(dt *A,dt *B,dt *C,int a,int b,int c);
void lsq(dt *left,dt *res,dt *right,dt *sample,int zuo,int rank,int you);
__global__ void gettt(dt *A,dt *B,dt *C,int m,int r);
void test(dt *A,dt *B,dt *C,int m,int r,int n);
void base(dt *left,dt *res,dt *right,int zuo,int rank,int you);

void basematrixpro(dt *A,dt *res,int row);
#endif
