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

void newmode(dt *A,dt *B,dt *sample,dt *left,dt *right,int m,int n,int r){
	// A is a random initial m*r, B is m*n ,sample is m*n
	// the we need A mul each col of sample 
	// and B.*sample
	// left is r*r*n  right is r*1*n store the data we want
//	cout<<"3"<<endl;
	dt *Bsample = new dt[m*n]();
	for(int i = 0;i<m*n;i++){
		Bsample[i] = sample[i]*B[i];
	}  
//	cout<<"3"<<endl;
	dt *Asample = new dt[m*r*n]();
	for(int i = 0;i<n;i++){
		for(int j = 0;j<r;j++){
			for(int k = 0;k<m;k++){
				Asample[i*m*r+j*m+k]=A[j*m+k]*sample[i*m+k];
			}
		//	Asample[i*m*r+j] = A[j]*sample[j%m+i*m];
		}
	}
//	cout<<"3"<<endl;
//	printTensor(Bsample,1,m,n);
//	printTensor(Asample,1,m*r,n);

	// now we have got Asample m*r*n and Bsample m*1*n
	// then we sent each front slice to GPU to get we want
	for(int i = 0;i<n;i++){
		// transpose product self and store in col
		signalpro(Asample+i*m*r,left+i*r*r,m,r);
		signalpro1(Asample+i*m*r,Bsample+i*m,right+i*r,m,r);
	}
//	printTensor(left,1,r*r,n);
//	printTensor(right,1,r*1,n);

//	cout<<"3"<<endl;
	delete[] Asample;Asample = nullptr;
	delete[] Bsample;Bsample = nullptr;
	
}
void signalpro(dt *A,dt *C,int a,int b){
	// A is a*b 
	dt *d_A;
	cudaMalloc((void**)&d_A,sizeof(dt)*a*b);
	dt *d_C;
	cudaMalloc((void**)&d_C,sizeof(dt)*b*b);
	dt alpha = 1.0;
	dt beta = 0.0;
	cudaMemcpy(d_A,A,sizeof(dt)*a*b,cudaMemcpyHostToDevice);

	cublasHandle_t handle;
	cublasCreate(&handle);
	cublasSgemm(
			handle,
			CUBLAS_OP_T,
			CUBLAS_OP_N,
			b,b,a,
			&alpha,
			d_A,a,
			d_A,a,
			&beta,
			d_C,b
			);
	cudaMemcpy(C,d_C,sizeof(dt)*b*b,cudaMemcpyDeviceToHost);
	cudaFree(d_A);
	cudaFree(d_C);
	cublasDestroy(handle);
}
void signalpro1(dt *A,dt *B,dt *C,int a,int b){
	//compute A'*B 
	// A is a*b , B is a*1 ,C is b*1
	dt *d_A;
	dt *d_B;
	cudaMalloc((void**)&d_A,sizeof(dt)*a*b);
	cudaMalloc((void**)&d_B,sizeof(dt)*a*1);
	dt *d_C;
	cudaMalloc((void**)&d_C,sizeof(dt)*b*1);
	dt alpha = 1.0;
	dt beta = 0.0;
	cudaMemcpy(d_A,A,sizeof(dt)*a*b,cudaMemcpyHostToDevice);
	cudaMemcpy(d_B,B,sizeof(dt)*a*1,cudaMemcpyHostToDevice);

	cublasHandle_t handle;
	cublasCreate(&handle);
	cublasSgemm(
			handle,
			CUBLAS_OP_T,
			CUBLAS_OP_N,
			b,1,a,
			&alpha,
			d_A,a,
			d_B,a,
			&beta,
			d_C,b
			);
	cudaMemcpy(C,d_C,sizeof(dt)*b*1,cudaMemcpyDeviceToHost);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	cublasDestroy(handle);

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
			CUBLAS_OP_T,
			CUBLAS_OP_T,
			a,
			c,
			b,
			&alpha,
			d_A,
			b,
			d_B,
			c,
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
