/*************************************************************************
	> File Name: trust.cpp
	> Author: hanlu
	> Mail: hanlu@shu.edu.cn 
	> Created Time: 2018年12月10日 星期一 21时13分39秒
 ************************************************************************/

#include "head.h"

int main(int argc,char *argv[]){

	int M =atoi(argv[1]);
	int N = M;
	int R = int(0.01*M);
    
	int count_device = 0;
    	cudaGetDeviceCount(&count_device);
    	if(count_device == 0){

    		printf("There is no GPUs\n");
	    	return false;
    	}
    	printf("There are %d GPUs\n",count_device);
    	for(int i=0;i<count_device;i++){
		cudaDeviceProp prop;
	    	if(cudaGetDeviceProperties(&prop,i) == cudaSuccess){
			printf("GPU %d: \t %s \n",i,prop.name);
	    	}
    	}
	int gpu_0 = 0;
	int gpu_1 = 1;

	srand((unsigned)time(NULL));
	dt *A = new dt[M*R]();
	for(int i = 0;i<M*R;i++){
		A[i] = rand()*0.1/(RAND_MAX*0.1);
	}
	dt *B = new dt[N*R]();
	for(int i = 0;i<N*R;i++){
		B[i] = rand()*0.1/(RAND_MAX*0.1);
	}

	dt *D = new dt[M*N]();
	matrixProduct(A,B,D,M,R,N); // D is the origin data
//	cout<<"origin data"<<endl;
//	printTensor(D,1,M*N,1);   // we view it store as col
	delete[] A;A=nullptr;
	delete[] B;B=nullptr;



// get transpose of D  N*M
	dt *D1 = new dt[M*N]();
	for(int i = 0;i<M;i++){
		for(int j = 0;j<N;j++){
			D1[i*N+j] = D[j*M+i];
		}
	}
//	printTensor(D1,1,M*N,1);

//	dt Omega[M*N]={1,0,1,1,0,1,1,0,1,1,0,1};
	dt *Omega = new dt[M*N]();
	for(int i = 0;i<M*N;i++){
		Omega[i] = rand()*0.1/(RAND_MAX*0.1);
		if(Omega[i]<=0.1){
			Omega[i] = 1;
		}else{
			Omega[i] = 0;
		}
	}
	cout<<"sample is 0.5"<<endl;

//	printTensor(Omega,1,M*N,1);
	dt *Omega1 = new dt[M*N]();
	for(int i = 0;i<M;i++){
		for(int j = 0;j<N;j++){
			Omega1[i*N+j] = Omega[j*M+i];
		}
	}
//	printTensor(Omega1,1,M*N,1);

//	dt U[M*R] = {3,6,2,6,2,6,1,8};
	dt *U = new dt[M*R]();
	for(int i = 0;i<M*R;i++){
		U[i] = rand()*0.1/(RAND_MAX*0.1);
	}

	dt *V = new dt[R*N]();

	dt *left = new dt[R*R*N]();
	dt *right = new dt[R*1*N]();
	dt *left1 = new dt[R*R*M]();
	dt *right1 = new dt[R*1*M]();

	dt *PB = new dt[M*K]();   //public data size of M*R store as column
	for(int i = 0;i<M*K;i++){
		PB[i] = D[i];
	}
	
	dt *MM = new dt[M*N]();  //MM is encpypted matrix
	dt *MM1 = new dt[M*N]();  //MM1 is its transpose

	Enc(D,Omega,PB,MM,MM1,M,N);
/* MM is incom encryped matrix of size m*n
   Omega is sample matrix
   we divide n into two part and each solve on one gpu   
*/
	int size0 = N/2;
	int size1 = M/2;
    double start, end;
    start = omp_get_wtime();

	for(int i = 0;i<10;i++){
	
		cudaSetDevice(gpu_0);
		getbatch(U,MM,Omega,left,right,M,size0,R);
		lsq(left,right,V,M,size0,R);

		cudaSetDevice(gpu_1);
		getbatch(U,MM+M*size0,Omega+M*size0,left+R*R*size0,right+R*size0,M,N-size0,R);
		lsq(left+R*R*size0,right+R*size0,V+R*size0,M,N-size0,R);
		Matrixtranpose(V,N,R);
		cudaDeviceSynchronize();
		
		cudaSetDevice(gpu_0);
		getbatch(V,MM1,Omega1,left1,right1,N,size1,R);
		lsq(left1,right1,U,N,size1,R);

		cudaSetDevice(gpu_1);
		getbatch(V,MM1+N*size1,Omega1+N*size1,left1+R*R*size1,right1+R*size1,N,M-size1,R);
		lsq(left1+R*R*size1,right1+R*size1,U+R*size1,N,M-size1,R);
		Matrixtranpose(U,M,R);
		cudaDeviceSynchronize();

	}
	// we need to U*V
	dt *result1 = new dt[M*N]();
	mproduct(U,V,result1,M,R,N);
    end = omp_get_wtime();
    double time = end-start;
    cout<<time<<endl;
	dt *result = new dt[M*N]();
	Dec(result1,result,PB,M,N); // decrypt result1 to result

	delete[] left;left=nullptr;
	delete[] right;right=nullptr;
	delete[] left1;left1=nullptr;
	delete[] right1;right1=nullptr;
	delete[] Omega1;Omega1=nullptr;
	delete[] Omega;Omega=nullptr;
	delete[] D1;D1=nullptr;
	delete[] U;U=nullptr;
	delete[] V;V=nullptr;
//	printTensor(result,1,M*N,1);
	

	double sh = 0.0;
	double xia = 0.0;
	for(int i = 0;i<M*N;i++){
		sh+=(result[i]-D[i])*(result[i]-D[i]);
		xia+=(D[i]*D[i]);
	}
	delete[] result;result=nullptr;
	delete[] result1;result1=nullptr;
	delete[] D;D=nullptr;
	delete[] MM;MM=nullptr;
	delete[] MM1;MM1=nullptr;
	double error=0.0;
	error = sqrt(sh)/sqrt(xia);
	cout<<"error is  "<<error<<endl;

	return 0;
}

