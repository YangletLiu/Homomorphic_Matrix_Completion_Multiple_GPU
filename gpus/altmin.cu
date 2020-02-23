#include "head.h"

void lsq(dt *left,dt *right,dt *res,int m,int n,int r){
	//left is r*r*n res is r*1*n
	//   r*r*batch      r*nrhs*batch
	// right  is r*1*n
	magma_init();
	magma_queue_t queue = NULL;
	magma_int_t dev = 0;
	magma_getdevice(&dev);
	magma_queue_create(dev,&queue);
	
	dt *d_left, *d_right; //origin data

	magma_int_t *d_ipiv;
	magma_int_t **dipiv_array;

	magma_int_t row, nrhs, lddl, lddr,batch;

	magmaFloat_ptr *darray_left;
	magmaFloat_ptr *darray_right;  //transfered

	magma_int_t *dinfo_array;

	row = r;
	nrhs = 1;
	batch = n;
	
	lddl = magma_roundup(row,32);
	lddr = magma_roundup(row,32);
	
	magma_smalloc(&d_left,lddl*r*batch);
	magma_smalloc(&d_right,lddr*1*batch);
	magma_imalloc(&d_ipiv,r*batch);
	magma_imalloc(&dinfo_array,batch);
	
	magma_malloc((void**)&darray_left, batch*sizeof(dt*));
	magma_malloc((void**)&darray_right, batch*sizeof(dt*));
	magma_malloc((void**)&dipiv_array, batch*sizeof(magma_int_t*));

	magma_ssetmatrix(r,r*batch,left,r,d_left,lddl,queue);
	magma_ssetmatrix(r,1*batch,right,r,d_right,lddr,queue);

	magma_sset_pointer(darray_left,d_left,lddl,0,0,lddl*r,batch,queue);
	magma_sset_pointer(darray_right,d_right,lddr,0,0,lddr,batch,queue);
	magma_iset_pointer(dipiv_array,d_ipiv,1,0,0,r,batch,queue);

	magma_sgesv_batched(
			row,
			nrhs,
			darray_left,
			lddl,
			dipiv_array,
			darray_right,
			lddr,
			dinfo_array,
			batch,
			queue
			);
	int *h_info = (int*)malloc(sizeof(int)*batch);
	magma_igetmatrix(batch,1,dinfo_array,batch,h_info,batch,queue);

//	for(int i = 0;i<batch;i++){
//		printf("%d ",h_info[i]);
//	}
//	printf("\n");
	magma_sgetmatrix(r,batch*1,d_right,lddr,res,r,queue);
	
//	printTensor(res,1,r*batch,1);
	
	free(h_info);
	magma_queue_destroy(queue);
	magma_free(d_left);
	magma_free(d_right);
	magma_free(darray_left);
	magma_free(darray_right);
	magma_free(d_ipiv);
	magma_free(dipiv_array);
	magma_free(dinfo_array);
	magma_finalize();

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
	cudaFree(d_res);
	cudaFree(d_res1);


}

