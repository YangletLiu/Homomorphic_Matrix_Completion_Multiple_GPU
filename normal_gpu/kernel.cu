#include "head.h"

__global__ void gettt(dt *A,dt *B,dt *C,int m,int r){
	
	int i = blockDim.x*blockIdx.x+threadIdx.x;
	const int temp = blockDim.x*gridDim.x;
	while(i<r*m){
		int row = i/m;
		int col = i%m;
		C[row*m+col] = B[col]*A[row*m+col];
		i+=temp;
	}
	__syncthreads();
}

void test(dt *A,dt *B,dt *C,int m,int r,int n){
	dt *d_A;
	cudaMalloc((void**)&d_A,sizeof(dt)*m*r);
	cudaMemcpy(d_A,A,sizeof(dt)*m*r,cudaMemcpyHostToDevice);
		dt *t3 = new dt[m*r]();
		cudaMemcpy(t3,d_A,sizeof(dt)*m*r,cudaMemcpyDeviceToHost);
		cout<<"hhk"<<endl;
		printTensor(t3,r,m,1);
		delete[] t3;t3=nullptr;

	for(int i = 0;i<n;i++){
	
		dt *t2 = new dt[m*r]();
		cudaMemcpy(t2,d_A,sizeof(dt)*m*r,cudaMemcpyDeviceToHost);
		cout<<"kkk"<<endl;
		printTensor(t2,r,m,1);
		delete[] t2;t2=nullptr;

		dt *d_tt;
		cudaMalloc((void**)&d_tt,sizeof(dt)*m*r);
		
		dt *d_sample;
		cudaMalloc((void**)&d_sample,sizeof(dt)*m);
		cudaMemcpy(d_sample,B+i*m,sizeof(dt)*m,cudaMemcpyHostToDevice);
		dt *t1 = new dt[m]();
		cudaMemcpy(t1,d_sample,sizeof(dt)*m,cudaMemcpyDeviceToHost);
		printTensor(t1,m,1,1);
		delete[] t1;t1=nullptr;

		dim3 th(512,1,1);
		dim3 bl((m*r+512-1)/512,1,1);
		gettt<<<bl,th>>>(d_A,d_sample,d_tt,m,r);

		dt *tt = new dt[m*r]();
		cudaMemcpy(tt,d_tt,sizeof(dt)*m*r,cudaMemcpyDeviceToHost);
		printTensor(tt,r,m,1);
		delete[] tt;tt=nullptr;

		cudaFree(d_tt);
		cudaFree(d_sample);

	}
	cudaFree(d_A);
}
