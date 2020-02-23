/*************************************************************************
	> File Name: opera.h
	> Author: hanlu
	> Mail: hanlu@shu.edu.cn 
	> Created Time: 2018年12月10日 星期一 21时14分15秒
 ************************************************************************/

#include "head.h"

void printTensor(dt *A,int a,int b,int c){
	for(int i = 0;i<c;i++){
		for(int j = 0;j<a;j++){
			for(int k =0;k<b;k++){
				cout<<A[i*a*b+j*b+k]<<"  ";
			}
			cout<<endl;
		}
		cout<<"-----------------------------------"<<endl;
	}
	cout<<endl;
}

void matrixProduct(dt *A,dt *B,dt *C,int a,int b,int c){
	// A is a*b; B is b*c  C is a*c 
	dt *d_A;
	dt *d_B;
	cudaMalloc((void**)&d_A,sizeof(dt)*a*b);
	cudaMalloc((void**)&d_B,sizeof(dt)*b*c);
	dt *d_C;
	cudaMalloc((void**)&d_C,sizeof(dt)*a*c);
	dt alpha = 1.0;
	dt beta = 0.0;
	cudaMemcpy(d_A,A,sizeof(dt)*a*b,cudaMemcpyHostToDevice);
	cudaMemcpy(d_B,B,sizeof(dt)*b*c,cudaMemcpyHostToDevice);

	cublasHandle_t handle;
	cublasCreate(&handle);
	cublasSgemm(
			handle,
			CUBLAS_OP_N,
			CUBLAS_OP_N,
			a,
			c,
			b,
			&alpha,
			d_A,
			a,
			d_B,
			b,
			&beta,
			d_C,  
			a
			);
	cudaMemcpy(C,d_C,sizeof(dt)*a*c,cudaMemcpyDeviceToHost);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	cublasDestroy(handle);
}


void mproduct(dt *A,dt *B,dt *C,int a,int b,int c){
	dt *d_A = NULL;	
	dt *d_B = NULL;	
	dt *d_C = NULL;
	cudaMalloc((void**)&d_A,sizeof(dt)*a*b); 	//a*b	
	cudaMalloc((void**)&d_B,sizeof(dt)*b*c);	//b*c	
	cudaMalloc((void**)&d_C,sizeof(dt)*a*c);	//a*c
	cudaMemcpy(d_A,A,sizeof(dt)*a*b,cudaMemcpyHostToDevice);
	cudaMemcpy(d_B,B,sizeof(dt)*b*c,cudaMemcpyHostToDevice);
	dt alpha = 1.0;
	dt beta = 0.0;
	cublasHandle_t handle;
	cublasCreate(&handle);
	cublasSgemm(
		handle,
		CUBLAS_OP_N,
		CUBLAS_OP_T,
		a,
		c,
		b,
		&alpha,
		d_A,
		a,
		d_B,
		c,
		&beta,
		d_C,
		a
	    );

	cudaMemcpy(C,d_C,sizeof(dt)*c*a,cudaMemcpyDeviceToHost);
	cublasDestroy(handle);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}
