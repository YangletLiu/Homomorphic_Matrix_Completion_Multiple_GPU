/*************************************************************************
	> File Name: trust.cpp
	> Author: hanlu
	> Mail: hanlu@shu.edu.cn 
	> Created Time: 2018年12月10日 星期一 21时13分39秒
 ************************************************************************/

#include "head.h"

int main(int argc,char *argv[]){

int luhan[11]={1000,222,500,1000,1500,2000,2500,3000,3500,4000,4500};
for(int hh=0;hh<11;hh++){
	int M = luhan[hh];
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
	int pl = N/2;    // divide MM M*N into pl and pr
	int pr = N-N/2;
	int ptl = M/2;   // divide MM1 N*M into ptl and ptr
	int ptr = M-M/2;
    
	double start, end;
    	start = omp_get_wtime();
// malloc  dta part to device 0
	cudaSetDevice(gpu_0);
	dt *dp_MM,*dp_MM1,*dp_U,*dp_V,*dp_Omega,*dp_Omega1;
	dt *dp_left,*dp_right,*dp_left1,*dp_right1;
	cudaMalloc((void**)&dp_MM,sizeof(dt)*M*pl);
	cudaMalloc((void**)&dp_MM1,sizeof(dt)*N*ptl);
	cudaMalloc((void**)&dp_U,sizeof(dt)*M*R);
	cudaMalloc((void**)&dp_V,sizeof(dt)*R*N);
	cudaMalloc((void**)&dp_Omega,sizeof(dt)*M*pl);
	cudaMalloc((void**)&dp_Omega1,sizeof(dt)*N*ptl);
	cudaMalloc((void**)&dp_left,sizeof(dt)*R*R*pl);
	cudaMalloc((void**)&dp_right,sizeof(dt)*R*1*pl);
	cudaMalloc((void**)&dp_left1,sizeof(dt)*R*R*ptl);
	cudaMalloc((void**)&dp_right1,sizeof(dt)*R*1*ptl);

// malloc  dta part to device 1
	cudaSetDevice(gpu_1);
	dt *dq_MM,*dq_MM1,*dq_U,*dq_V,*dq_Omega,*dq_Omega1;
	dt *dq_left,*dq_right,*dq_left1,*dq_right1;
	cudaMalloc((void**)&dq_MM,sizeof(dt)*M*pr);
	cudaMalloc((void**)&dq_MM1,sizeof(dt)*N*ptr);
	cudaMalloc((void**)&dq_U,sizeof(dt)*M*R);
	cudaMalloc((void**)&dq_V,sizeof(dt)*R*N);
	cudaMalloc((void**)&dq_Omega,sizeof(dt)*M*pr);
	cudaMalloc((void**)&dq_Omega1,sizeof(dt)*N*ptr);
	cudaMalloc((void**)&dq_left,sizeof(dt)*R*R*pr);
	cudaMalloc((void**)&dq_right,sizeof(dt)*R*1*pr);
	cudaMalloc((void**)&dq_left1,sizeof(dt)*R*R*ptr);
	cudaMalloc((void**)&dq_right1,sizeof(dt)*R*1*ptr);

// transfer data to GPU
	cudaSetDevice(gpu_0);
	cudaMemcpy(dp_MM,MM,sizeof(dt)*M*pl,cudaMemcpyHostToDevice);
	cudaMemcpy(dp_MM1,MM1,sizeof(dt)*N*ptl,cudaMemcpyHostToDevice);
	cudaMemcpy(dp_U,U,sizeof(dt)*M*R,cudaMemcpyHostToDevice);
	cudaMemcpy(dp_V,V,sizeof(dt)*R*N,cudaMemcpyHostToDevice);
	cudaMemcpy(dp_Omega,Omega,sizeof(dt)*M*pl,cudaMemcpyHostToDevice);
	cudaMemcpy(dp_Omega1,Omega1,sizeof(dt)*N*ptl,cudaMemcpyHostToDevice);
// transfer data to GPU
	cudaSetDevice(gpu_1);
	cudaMemcpy(dq_MM,MM+M*pl,sizeof(dt)*M*pr,cudaMemcpyHostToDevice);
	cudaMemcpy(dq_MM1,MM1+N*ptl,sizeof(dt)*N*ptr,cudaMemcpyHostToDevice);
	cudaMemcpy(dq_U,U,sizeof(dt)*M*R,cudaMemcpyHostToDevice);
	cudaMemcpy(dq_V,V,sizeof(dt)*R*N,cudaMemcpyHostToDevice);
	cudaMemcpy(dq_Omega,Omega+M*pl,sizeof(dt)*M*pr,cudaMemcpyHostToDevice);
	cudaMemcpy(dq_Omega1,Omega1+N*ptl,sizeof(dt)*N*ptr,cudaMemcpyHostToDevice);

// compute each part
	for(int i = 0;i<10;i++){
	
		cudaSetDevice(gpu_0);
		getbatch(dp_U,dp_MM,dp_Omega,dp_left,dp_right,M,pl,R);
		lsq(dp_left,dp_right,dp_V,M,pl,R);
		cudaMemcpyPeerAsync(dq_V,gpu_1,dp_V,gpu_0,sizeof(dt)*R*pl);

		cudaSetDevice(gpu_1);
		getbatch(dq_U,dq_MM,dq_Omega,dq_left,dq_right,M,pr,R);
		lsq(dq_left,dq_right,dq_V+R*pl,M,pr,R);
		cudaMemcpyPeerAsync(dp_V+R*pl,gpu_0,dq_V+R*pl,gpu_1,sizeof(dt)*R*pr);
		cudaDeviceSynchronize();

		cudaSetDevice(gpu_0);
		Matrixtranpose(dp_V,gpu_0,N,R);
		cudaSetDevice(gpu_1);
		Matrixtranpose(dq_V,gpu_1,N,R);
		cudaDeviceSynchronize();
		
		cudaSetDevice(gpu_0);
		getbatch(dp_V,dp_MM1,dp_Omega1,dp_left1,dp_right1,N,ptl,R);
		lsq(dp_left1,dp_right1,dp_U,N,ptl,R);
		cudaMemcpyPeerAsync(dq_U,gpu_1,dp_U,gpu_0,sizeof(dt)*R*ptl);

		cudaSetDevice(gpu_1);
		getbatch(dq_V,dq_MM1,dq_Omega1,dq_left1,dq_right1,N,ptr,R);
		lsq(dq_left1,dq_right1,dq_U+R*ptl,N,ptr,R);
		cudaMemcpyPeerAsync(dp_U+R*ptl,gpu_0,dq_U+R*ptl,gpu_1,sizeof(dt)*R*ptr);
		cudaDeviceSynchronize();

		cudaSetDevice(gpu_0);
		Matrixtranpose(dp_U,gpu_0,M,R);
		cudaSetDevice(gpu_1);
		Matrixtranpose(dq_U,gpu_1,M,R);
		cudaDeviceSynchronize();

	}
	// we need to U*V
	dt *result1 = new dt[M*N]();
	mproduct(dq_U,dq_V,result1,M,R,N);
	cudaDeviceSynchronize();

	cudaSetDevice(gpu_0);
	cudaFree(dp_MM); cudaFree(dp_MM1); cudaFree(dp_U);
	cudaFree(dp_V); cudaFree(dp_Omega); cudaFree(dp_Omega1);
	cudaFree(dp_left); cudaFree(dp_right); cudaFree(dp_left1);
	cudaFree(dp_right1);
    
	cudaSetDevice(gpu_1);
	cudaFree(dq_MM); cudaFree(dq_MM1); cudaFree(dq_U);
	cudaFree(dq_V); cudaFree(dq_Omega); cudaFree(dq_Omega1);
	cudaFree(dq_left); cudaFree(dq_right); cudaFree(dq_left1);
	cudaFree(dq_right1);

    	end = omp_get_wtime();
    	double time = end-start;
    	cout<<time<<endl;

	dt *result = new dt[M*N]();
	Dec(result1,result,PB,M,N); // decrypt result1 to result

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
	ofstream outfile("v100q6000.txt",ios::app);
	outfile<<M<<"*"<<N<<" "<<R<<" ";
//	outfile<<(double)(t2-t1)/CLOCKS_PER_SEC<<"s"<<"  ";
	outfile<<time<<"s"<<"  ";
	outfile<< error<<endl;
	outfile.close();
}
	return 0;
}

