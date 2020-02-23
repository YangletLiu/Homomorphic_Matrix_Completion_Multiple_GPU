#include "head.h"

void base(dt *left,dt *res,dt *right,int zuo,int rank,int you){
	
	dt *d_left;
	cudaMalloc((void**)&d_left,sizeof(dt)*zuo*rank);
	cudaMemcpy(d_left,left,sizeof(dt)*zuo*rank,cudaMemcpyHostToDevice);
	dt *d_right;
	cudaMalloc((void**)&d_right,sizeof(dt)*zuo);
	cudaMemcpy(d_right,right,sizeof(dt)*zuo,cudaMemcpyHostToDevice);

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
	cudaMalloc((void**)&d_tau,sizeof(dt)*zuo);

	cusolverDnSgeqrf_bufferSize(
			cusolverH,
			zuo,rank,
			d_left,zuo,
			&lwork
			);
	cudaMalloc((void**)&d_work,sizeof(dt)*lwork);
	cusolverDnSgeqrf(
			cusolverH,
			zuo,rank,
			d_left,zuo,
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
			zuo,1,rank,
			d_left,zuo,
			d_tau,
			d_right,zuo,
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
			rank,1,
			&one,
			d_left,zuo,
			d_right,rank
			);
	cudaDeviceSynchronize();
	cudaMemcpy(res,d_right,sizeof(dt)*rank,cudaMemcpyDeviceToHost);
	
	cudaFree(d_tau);
	cudaFree(d_work);
	cudaFree(d_left);
	cudaFree(d_right);
	cudaFree(devInfo);
	cublasDestroy(cublasH);
	cusolverDnDestroy(cusolverH);
}
void lsq(dt *left,dt *res,dt *right,dt *sample,int zuo,int rank,int you){
	// right is m*n with unknown is set 0
	// left is m*r res is r*n 
	// sample 
	//step1 left = QR
	dt *d_left;
	cudaMalloc((void**)&d_left,sizeof(dt)*zuo*rank);
	cudaMemcpy(d_left,left,sizeof(dt)*zuo*rank,cudaMemcpyHostToDevice);

	//initial jubin
	cusolverDnHandle_t cusolverH = NULL;
	cublasHandle_t cublasH = NULL;
	cusolverDnCreate(&cusolverH);
	cublasCreate(&cublasH);
	
for(int i = 0;i<you;i++){
	dt *d_temp;
	cudaMalloc((void**)&d_temp,sizeof(dt)*zuo);
	cudaMemcpy(d_temp,right+i*zuo,sizeof(dt)*zuo,cudaMemcpyHostToDevice);

	dt *d_sample;
	cudaMalloc((void**)&d_sample,sizeof(dt)*zuo);
	cudaMemcpy(d_sample,sample+i*zuo,sizeof(dt)*zuo,cudaMemcpyHostToDevice);

	dt *d_tt;
	cudaMalloc((void**)&d_tt,sizeof(dt)*zuo*rank);

	dim3 thread(512,1,1);
	dim3 block((zuo*rank+512-1)/512,1,1);
	gettt<<<block,thread>>>(d_left,d_sample,d_tt,zuo,rank);
//	cout<<zuo<<"  "<<you<<endl;
/*	dt *t1 = new dt[zuo*rank]();
	cudaMemcpy(t1,d_tt,sizeof(dt)*zuo*rank,cudaMemcpyDeviceToHost);
	printTensor(t1,1,zuo*rank,1);
	delete[] t1;t1=nullptr;

	dt *t2 = new dt[zuo]();
	cudaMemcpy(t2,d_temp,sizeof(dt)*zuo,cudaMemcpyDeviceToHost);
	printTensor(t2,1,zuo,1);
	delete[] t2;t2=nullptr;
*/

//	int info_gpu = 0;
	int *devInfo = NULL;
	cudaMalloc((void**)&devInfo,sizeof(int));
	dt *d_work = NULL;
	int lwork = 0;
	dt *d_tau = NULL;
	cudaMalloc((void**)&d_tau,sizeof(dt)*zuo);

	cusolverDnSgeqrf_bufferSize(
			cusolverH,
			zuo,rank,
			d_tt,zuo,
			&lwork
			);
	cudaMalloc((void**)&d_work,sizeof(dt)*lwork);
	cusolverDnSgeqrf(
			cusolverH,
			zuo,rank,
			d_tt,zuo,
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
			zuo,1,rank,
			d_tt,zuo,
			d_tau,
			d_temp,zuo,
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
			rank,1,
			&one,
			d_tt,zuo,
			d_temp,rank
			);
	cudaDeviceSynchronize();
	cudaMemcpy(res+i*rank,d_temp,sizeof(dt)*rank,cudaMemcpyDeviceToHost);
	
//	printTensor(res+i*rank,1,rank,1);
//	cout<<"ovber"<<endl;

	cudaFree(d_tau);
	cudaFree(d_work);
	cudaFree(d_temp);
	cudaFree(d_tt);
	cudaFree(d_sample);
	cudaFree(devInfo);

	
}
//	printTensor(res,n,r,1);
	// now res is n*r we want to transfer to r*n
	dt *d_res = NULL;	
	dt *d_res1 = NULL;
	cudaMalloc((void**)&d_res,sizeof(dt)*you*rank);	//a*c	
	cudaMalloc((void**)&d_res1,sizeof(dt)*you*rank);	//b*c
	cudaMemcpy(d_res,res,sizeof(dt)*you*rank,cudaMemcpyHostToDevice);
	dt alpha = 1.0;
	dt beta = 0.0;
	cublasHandle_t handle;
	cublasCreate(&handle);
	cublasSgeam(
		handle,
		CUBLAS_OP_T,
		CUBLAS_OP_N,
		you,
		rank,
		&alpha,
		d_res,
		rank,
		&beta,
		d_res1,
		you,
		d_res1,
		you
		 );
	cudaMemcpy(res,d_res1,sizeof(dt)*you*rank,cudaMemcpyDeviceToHost);
//	printTensor(res,1,n*r,1);

	cublasDestroy(cublasH);
	cusolverDnDestroy(cusolverH);
	cudaFree(d_left);
	cudaFree(d_res);
	cudaFree(d_res1);
	cudaDeviceReset();
}
