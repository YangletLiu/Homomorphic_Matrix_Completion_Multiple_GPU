#include "head.h"

void base(dt *A,dt *B,dt *sample,dt *V,int m,int n,int r){
	

	dt *d_sample;
	dt *d_Bsample;
	cudaMalloc((void**)&d_sample,sizeof(dt)*m*n);
	cudaMalloc((void**)&d_Bsample,sizeof(dt)*m*n);
	cudaMemcpy(d_sample,sample,sizeof(dt)*m*n,cudaMemcpyHostToDevice);
	cudaMemcpy(d_Bsample,B,sizeof(dt)*m*n,cudaMemcpyHostToDevice);

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
	// now left and right are in device memory
	// left is r*r*n right is r*1*n	

	cusolverDnHandle_t cusolverH = NULL;
	cublasHandle_t cublasH = NULL;
	cusolverDnCreate(&cusolverH);
	cublasCreate(&cublasH);

//	int info_gpu=0;
	int *devInfo = NULL;
	cudaMalloc((void**)&devInfo,sizeof(int));
	dt *d_work = NULL;
	int lwork = 0;
	dt *d_tau = NULL;
	cudaMalloc((void**)&d_tau,sizeof(dt)*r);

	for(int i = 0;i<n;i++){
	cusolverDnSgeqrf_bufferSize(
			cusolverH,
			r,r,
			d_left+i*r*r,r,
			&lwork
			);
	cudaMalloc((void**)&d_work,sizeof(dt)*lwork);
	cusolverDnSgeqrf(
			cusolverH,
			r,r,
			d_left+i*r*r,r,
			d_tau,
			d_work,
			lwork,
			devInfo
			);
	cudaDeviceSynchronize();
//	cudaMemcpy(&info_gpu,devInfo,sizeof(int),cudaMemcpyDeviceToHost);
//	cout<<info_gpu<<endl;

	//step2 compute Q^T*right

	cusolverDnSormqr(
			cusolverH,
			CUBLAS_SIDE_LEFT,
			CUBLAS_OP_T,
			r,1,r,
			d_left+r*r*i,r,
			d_tau,
			d_right+i*r*1,r,
			d_work,
			lwork,
			devInfo
			);
	cudaDeviceSynchronize();
//	cudaMemcpy(&info_gpu,devInfo,sizeof(int),cudaMemcpyDeviceToHost);
//	cout<<info_gpu<<endl;

	// step3 solve R*x = Q^T*b 
	dt one = 1;
	cublasStrsm(
			cublasH,
			CUBLAS_SIDE_LEFT,
			CUBLAS_FILL_MODE_UPPER,
			CUBLAS_OP_N,
			CUBLAS_DIAG_NON_UNIT,
			r,1,
			&one,
			d_left+i*r*r,r,
			d_right+i*r*1,r
			);
	cudaDeviceSynchronize();
	}
	cudaMemcpy(V,d_right,sizeof(dt)*r*n,cudaMemcpyDeviceToHost);
	
	cudaFree(d_tau);
	cudaFree(d_work);
	cudaFree(d_left);
	cudaFree(d_right);
	cudaFree(devInfo);
	cublasDestroy(cublasH);
	cusolverDnDestroy(cusolverH);

	dt *d_V = NULL;	
	dt *d_res1 = NULL;
	cudaMalloc((void**)&d_V,sizeof(dt)*r*n);	//a*c	
	cudaMalloc((void**)&d_res1,sizeof(dt)*r*n);	//b*c
	cudaMemcpy(d_V,V,sizeof(dt)*n*r,cudaMemcpyHostToDevice);
	cublasSgeam(
		handle,
		CUBLAS_OP_T,
		CUBLAS_OP_N,
		n,
		r,
		&alpha,
		d_V,
		r,
		&beta,
		d_res1,
		n,
		d_res1,
		n
		 );
	cudaMemcpy(V,d_res1,sizeof(dt)*n*r,cudaMemcpyDeviceToHost);

	cublasDestroy(handle);
	cudaFree(d_V);
	cudaFree(d_res1);

}
