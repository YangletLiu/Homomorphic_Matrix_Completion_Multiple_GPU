#include "head.h"

/*__global__ void elepro(dt *t1,dt *t2,dt *tt,int k){
	
	int i = blockDim.x*blockIdx.x+threadIdx.x;
	const int temp = blockDim.x*gridDim.x;
	while(i<k){
		tt[i] = t1[i]*t2[i];
		i+=temp;
	}
	__syncthreads();

}

__global__ void elebat(dt *t1,dt *t2,dt *tt,int a,int b,int c){
// m r n
	int i = blockDim.x*blockIdx.x+threadIdx.x;
	const int temp = blockDim.x*gridDim.x;
	while(i<a*b*c){
		int tube = i/(a*b); //locate which slice
		int row = (i-tube*(a*b))/a;
		int col = (i-tube*(a*b))%a;
		tt[tube*a*b+row*a+col] = t1[row*a+col]*t2[col+tube*a];
		i+=temp;
	}
	__syncthreads();	
}
*/
__global__ void fftResultProcess(float* d_t,const int num,const int len){
const int tid = blockIdx.x*blockDim.x+threadIdx.x;
if(tid < num){
	d_t[tid]=d_t[tid]/len;
	}
	__syncthreads();
}

__global__ void fftResultProcess(cuComplex* d_t,const int num, const int len){
	const int tid = blockIdx.x*blockDim.x+threadIdx.x;
	if(tid < num){
	d_t[tid].x=d_t[tid].x/len;
	__syncthreads();
	d_t[tid].y=d_t[tid].y/len; 
	}
	__syncthreads();	
}

__global__ void elepro(dt *t1,dt *t2,dt *t3,dt *tt,int m,int n){
	
	int i = blockDim.x*blockIdx.x+threadIdx.x;
	const int temp = blockDim.x*gridDim.x;
	while(i<m*n){
		int col = i/m;
		int row = i%m;	
		tt[col*m+row] =((t1[col*m+row]*t2[col*m+row])*0.5+0.1*t3[row]+0.1*t3[m+row]+0.1*t3[2*m+row]+0.1*t3[3*m+row]+0.1*t3[4*m+row])*t2[col*m+row];
		i+=temp;
	}
	__syncthreads();

}
__global__ void elepro1(dt *t1,dt *t2,dt *tt,int m,int n){
	
	int i = blockDim.x*blockIdx.x+threadIdx.x;
	const int temp = blockDim.x*gridDim.x;
	while(i<m*n){
		int col = i/m;
		int row = i%m;	
		tt[col*m+row] =((t1[col*m+row])-0.1*t2[row]-0.1*t2[m+row]-0.1*t2[2*m+row]-0.1*t2[3*m+row]-0.1*t2[4*m+row])/0.5;
		i+=temp;
	}
	__syncthreads();
}
__global__ void obt(dt *X,dt *sample,dt *res,int m,int r,int n){
	
	int i = blockDim.x*blockIdx.x+threadIdx.x;  // number of threads
	int slice = i/1024;   // lock to compute which slice
	int tix = i%1024;
	__shared__ float tmp[4500];   //each block has share memory store one col of sample
	for(int j = 0;j<m;j++){
		tmp[j]=sample[slice*m+j];	
	}   //  now each block has their own shared memory

	while(tix<m*r){
		int row = tix%m;
		int col = tix/m;
		res[slice*m*r+col*m+row]=X[col*m+row]*tmp[row];

		tix=tix+1024;
	}	
}
void Dec(dt *A,dt *B,dt *PB,int m,int n){
	// completed matrix A, B is decrypted matrix 
	// PB is publlic data
	int k = K;
	dt *d_A;
	dt *d_B;
	dt *d_PB;
	cudaMalloc((void**)&d_A,sizeof(dt)*m*n);
	cudaMalloc((void**)&d_B,sizeof(dt)*m*n);
	cudaMalloc((void**)&d_PB,sizeof(dt)*m*k);
	cudaMemcpy(d_A,A,sizeof(dt)*m*n,cudaMemcpyHostToDevice);
	cudaMemcpy(d_PB,PB,sizeof(dt)*m*k,cudaMemcpyHostToDevice);
	dim3 thread(512,1,1);
	dim3 block((m*n+512-1)/512,1,1);
	elepro1<<<block,thread>>>(d_A,d_PB,d_B,m,n);
	cudaMemcpy(B,d_B,sizeof(dt)*m*n,cudaMemcpyDeviceToHost);
	
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_PB);

}
void Enc(dt *D,dt *sample,dt *PB,dt *MM,dt *MM1,int m,int n){
	// D is origin matrix m*n sample is binary matrix m*n
	// PB is m*k is public data to encrypt
	// MM is encpypted matrix and MM1 is its transpose;
	int k = K;
	dt *d_D;
	dt *d_sample;
	dt *d_PB;

	dt *d_MM;
	cudaMalloc((void**)&d_D,sizeof(dt)*m*n);
	cudaMalloc((void**)&d_sample,sizeof(dt)*m*n);
	cudaMalloc((void**)&d_PB,sizeof(dt)*m*k);

	cudaMalloc((void**)&d_MM,sizeof(dt)*m*n);

	cudaMemcpy(d_sample,sample,sizeof(dt)*m*n,cudaMemcpyHostToDevice);
	cudaMemcpy(d_D,D,sizeof(dt)*m*n,cudaMemcpyHostToDevice);
	cudaMemcpy(d_PB,PB,sizeof(dt)*m*k,cudaMemcpyHostToDevice);
	dim3 thread(512,1,1);
	dim3 block((m*n+512-1)/512,1,1);
	elepro<<<block,thread>>>(d_D,d_sample,d_PB,d_MM,m,n);
	cudaMemcpy(MM,d_MM,sizeof(dt)*m*n,cudaMemcpyDeviceToHost);
     
	//m*n to n*m
	dt *d_MM1 = NULL;
	cudaMalloc((void**)&d_MM1,sizeof(dt)*m*n);	//b*c
	dt alpha = 1.0;
	dt beta = 0.0;
	cublasHandle_t handle1;
	cublasCreate(&handle1);
	cublasSgeam(
		handle1,
		CUBLAS_OP_T,
		CUBLAS_OP_N,
		m,
		n,
		&alpha,
		d_MM,
		n,
		&beta,
		d_MM1,
		m,
		d_MM1,
		m
		 );
	cudaMemcpy(MM1,d_MM1,sizeof(dt)*m*n,cudaMemcpyDeviceToHost);
	cudaFree(d_MM);
	cudaFree(d_MM1);
	cudaFree(d_D);
	cudaFree(d_sample);
	cudaFree(d_PB);
	cublasDestroy(handle1);

}

void getbatch(dt *A,dt *B,dt *sample,dt *left,dt *right,int m,int n,int r){
	// A is a random initial m*r, B is m*n, sample is m*n
	// the we need A mul each col of sample 
	// and B.*sample
	// left is r*r*n  right is r*1*n store the data we want
	dt *d_sample;
	dt *d_Bsample;
	cudaMalloc((void**)&d_sample,sizeof(dt)*m*n);
	cudaMalloc((void**)&d_Bsample,sizeof(dt)*m*n);
	cudaMemcpy(d_sample,sample,sizeof(dt)*m*n,cudaMemcpyHostToDevice);
	cudaMemcpy(d_Bsample,B,sizeof(dt)*m*n,cudaMemcpyHostToDevice);

/*	dt *temp = new dt[m*n]();
	cudaMemcpy(temp,d_Bsample,sizeof(dt)*m*n,cudaMemcpyDeviceToHost);
	printTensor(temp,1,m*n,1);
	delete[] temp;temp=nullptr;
*/
	dt *d_Asample;
	dt *d_A;
	cudaMalloc((void**)&d_A,sizeof(dt)*m*r);
	cudaMalloc((void**)&d_Asample,sizeof(dt)*m*r*n);
	cudaMemcpy(d_A,A,sizeof(dt)*m*r,cudaMemcpyHostToDevice);
	dim3 th2(1024,1,1);
	dim3 bl2(n,1,1);
	obt<<<bl2,th2>>>(d_A,d_sample,d_Asample,m,r,n);
	cudaFree(d_A);
	cudaFree(d_sample);

/*	dt *temp1 = new dt[m*r*n]();
	cudaMemcpy(temp1,d_Asample,sizeof(dt)*m*r*n,cudaMemcpyDeviceToHost);
	printTensor(temp1,1,m*r,n);
	delete[] temp1;temp1=nullptr;
*/
	// the we will compute batched product by cublasSgemmStridedBatched
	// A'*Ax = A'*b
	// d_Asample is m*r*n d_Bsample is m*1*n

	dt *d_left;
	cudaMalloc((void**)&d_left,sizeof(dt)*r*r*n);
	cublasHandle_t handle;
	cublasCreate(&handle);
	dt alpha = 1.0;
	dt beta = 0.0;
	cublasSgemmStridedBatched(
			handle,
			CUBLAS_OP_T,
			CUBLAS_OP_N,
			r,r,m,
			&alpha,
			d_Asample,m,m*r,
			d_Asample,m,m*r,
			&beta,
			d_left,r,r*r,
			n
			);
	
	cudaMemcpy(left,d_left,sizeof(dt)*r*r*n,cudaMemcpyDeviceToHost);
//	printTensor(left,1,r*r,n);
	
	dt *d_right;
	cudaMalloc((void**)&d_right,sizeof(dt)*r*1*n);
	cublasSgemmStridedBatched(
			handle,
			CUBLAS_OP_T,
			CUBLAS_OP_N,
			r,1,m,
			&alpha,
			d_Asample,m,m*r,
			d_Bsample,m,m,
			&beta,
			d_right,r,r,
			n
			);

	cudaMemcpy(right,d_right,sizeof(dt)*r*1*n,cudaMemcpyDeviceToHost);
//	printTensor(right,1,r*1,n);
	
	cudaFree(d_Asample);
	cudaFree(d_Bsample);
	cudaFree(d_left);
	cudaFree(d_right);

	cublasDestroy(handle);

}
