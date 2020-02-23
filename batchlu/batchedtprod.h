#ifndef _BATCHEDTPROD_H_
#define _BATCHEDTPROD_H_
#include  "based.h"
#include <cufft.h>
#include <cublas_v2.h>
void batchedtprod(float* t1,float* t2,float* T,cublasOperation_t t_t1,cublasOperation_t t_t2,int row, int col, int rank, int tupe);
#endif
