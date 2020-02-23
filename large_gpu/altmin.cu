#include "head.h"

void lsq(dt *left,dt *right,dt *res,int m,int n,int r){
	//left is r*r*n res is r*1*n V is r*1*n
	// step1 point array replace old 
	cusolverDnHandle_t handle = NULL;
	cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
	cusolverDnCreate(&handle);

//	int *infoArray = new int[n]();
	int *d_infoArray = NULL;
	cudaMalloc((void**)&d_infoArray,sizeof(int)*n);     //check flag

	dt **h_left = new dt*[n];
	dt **h_right = new dt*[n];

	// transfer data to GPU with pointarray
	for(int i = 0;i<n;i++){
		cudaMalloc((void**)&h_left[i],sizeof(dt)*r*r);
		cudaMemcpy(h_left[i],left+i*r*r,sizeof(dt)*r*r,cudaMemcpyHostToDevice);
		cudaMalloc((void**)&h_right[i],sizeof(dt)*r*1);		
		cudaMemcpy(h_right[i],right+i*r*1,sizeof(dt)*r*1,cudaMemcpyHostToDevice);
	}  
	cudaDeviceSynchronize();

	dt **d_left; 
	dt **d_right;
	cudaMalloc((void**)&d_left,sizeof(dt*)*n);
	cudaMalloc((void**)&d_right,sizeof(dt*)*n);

	cudaMemcpy(d_left,h_left,sizeof(dt*)*n,cudaMemcpyHostToDevice);
	cudaMemcpy(d_right,h_right,sizeof(dt*)*n,cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();

	cusolverDnSpotrfBatched(
		handle,
		uplo,
		r,
		d_left,r,
		d_infoArray,
		n
		);
	cudaDeviceSynchronize();
/*	cudaMemcpy(infoArray,d_infoArray,sizeof(int)*n,cudaMemcpyDeviceToHost);
	for(int j = 0;j<n;j++){
		cout<<infoArray[j]<<"  ";
	}
	cout<<endl;
*/
	cudaDeviceSynchronize();
	
	cusolverDnSpotrsBatched(
		handle,
		uplo,
		r,1,
		d_left,r,
		d_right,r,
		d_infoArray,
		n
		);
	cudaDeviceSynchronize();
/*	cudaMemcpy(infoArray,d_infoArray,sizeof(int)*n,cudaMemcpyDeviceToHost);
	for(int j = 0;j<n;j++){
		cout<<infoArray[j]<<"  ";
	}
	cout<<endl;
*/
	for(int i = 0;i<n;i++){
		cudaMemcpy(res+i*r*1,h_right[i],sizeof(dt)*r*1,cudaMemcpyDeviceToHost);
		printTensor(res+i*r*1,1,r,1);
	}
	cudaDeviceSynchronize();

	cudaFree(d_left);
	cudaFree(d_right);
	cudaFree(d_infoArray);
	delete[] h_left;h_left = nullptr;
	delete[] h_right;h_right = nullptr;
//	delete[] infoArray;infoArray=nullptr;

	//printTensor(res,n,r,1);
	// now res is n*r we want to transfer to r*n
	dt *d_res = NULL;	
	dt *d_res1 = NULL;
	cudaMalloc((void**)&d_res,sizeof(dt)*r*n);	//a*c	
	cudaMalloc((void**)&d_res1,sizeof(dt)*r*n);	//b*c
	cudaMemcpy(d_res,res,sizeof(dt)*n*r,cudaMemcpyHostToDevice);
	dt alpha = 1.0;
	dt beta = 0.0;
	cublasHandle_t handle1;
	cublasCreate(&handle1);
	cublasSgeam(
		handle1,
		CUBLAS_OP_T,
		CUBLAS_OP_N,
		n,
		r,
		&alpha,
		d_res,
		r,
		&beta,
		d_res1,
		n,
		d_res1,
		n
		 );
	cudaMemcpy(res,d_res1,sizeof(dt)*n*r,cudaMemcpyDeviceToHost);

	cublasDestroy(handle1);
	cusolverDnDestroy(handle);
	cudaFree(d_res);
	cudaFree(d_res1);
	cudaDeviceReset();

}

