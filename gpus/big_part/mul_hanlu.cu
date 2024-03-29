/*************************************************************************
	> File Name: trust.cpp
	> Author: hanlu
	> Mail: hanlu@shu.edu.cn 
	> Created Time: 2018年12月10日 星期一 21时13分39秒
 ************************************************************************/

#include "head.h"

int main(int argc,char *argv[]){
	int luhan[13] = {4003,7000,5000,6000,7000,8000,9000,10000,11000,12000,13000,14000,15000};
	for(int hh =0;hh<13;hh++){
    double start, end;
    start = omp_get_wtime();
	int M = luhan[hh];
	int N = M;
	int R;
	int T;
//	cout<<M<<endl;	
	if(M>=N){
	 R = int(M*0.01)+1;
	}else{
	 R = int(N*0.01)+1;
	}
	double real = (M/4500.0)*(N/4500.0)*(R/45.0);
	int rr = int(real);
//	cout<<real<<endl;
	if((real-rr)>0){
		T = rr+1;
	}else{
		T = rr;
	}
//	cout<<T<<endl;
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
//	cout<<"origin is"<<endl;
//	printTensor(D,1,M*N,1);
	delete[] A;A=nullptr;
	delete[] B;B=nullptr;

	// get transpose of D  N*M
	dt *D1 = new dt[M*N]();
	for(int i = 0;i<M;i++){
		for(int j = 0;j<N;j++){
			D1[i*N+j] = D[j*M+i];
		}
	}

	dt *Omega = new dt[M*N]();
	for(int i = 0;i<M*N;i++){
		Omega[i] = rand()*0.1/(RAND_MAX*0.1);
		if(Omega[i]<=0.5){
			Omega[i] = 1;
		}else{
			Omega[i] = 0;
		}
	}
//	dt Omega[M*N]={1,0,1,1,0,1,1,0,1,1,0,1};

//	cout<<"sample is"<<endl;
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
	dt *result = new dt[M*N]();	

	if(T>1){
		int Tp = T/2;    //part for V100 gpu
		int Tq = T-Tp;   //part for q6000 gpu
		cout<<Tp<<"  "<<Tq<<endl; 

		int Nt,Mt;
		if (N%T!=0){
		 Nt = int(N/T)+1;
		}else{
			Nt = N/T;
		}
		if (M%T!=0){
		 Mt = int(M/T)+1;
		}else{
			Mt = M/T;
		}

		int yichu,yichu1;
		yichu = N-(T-1)*Nt;
		yichu1 = M-(T-1)*Mt;
		dt *last_left = new dt[R*R*yichu]();
		dt *last_right = new dt[R*1*yichu]();
		dt *left = new dt[R*R*Nt]();
		dt *right = new dt[R*1*Nt]();
		dt *left1 = new dt[R*R*Mt]();
		dt *right1 = new dt[R*1*Mt]();
		dt *last_left1 = new dt[R*R*yichu1]();
		dt *last_right1 = new dt[R*1*yichu1]();

	for(int i = 0;i<10;i++){
		cudaSetDevice(0);
		for(int t = 0;t<Tp;t++){
			getbatch(U,D+M*Nt*t,Omega+M*Nt*t,left,right,M,Nt,R);	
			lsq(left,right,V+R*Nt*t,M,Nt,R);
		}
		cudaSetDevice(1);
		for(int t = 0;t<Tq;t++){
			if(t==(Tq-1)){
				getbatch(U,D+M*Nt*(t+Tp),Omega+M*Nt*(t+Tp),last_left,last_right,M,yichu,R);
				lsq(last_left,last_right,V+R*Nt*(t+Tp),M,yichu,R);
			}else{
				getbatch(U,D+M*Nt*(t+Tp),Omega+M*Nt*(t+Tp),left,right,M,Nt,R);	
				lsq(left,right,V+R*Nt*(t+Tp),M,Nt,R);
			}
		}

		cudaDeviceSynchronize();
		Matrixtranpose(V,N,R);

		cudaSetDevice(0);
		for(int t = 0;t<Tp;t++){
			getbatch(V,D1+N*Mt*t,Omega1+N*Mt*t,left1,right1,N,Mt,R);
			lsq(left1,right1,U+R*Mt*t,N,Mt,R);
		}

		cudaSetDevice(1);
		for(int t = 0;t<Tq;t++){
			if(t==(Tq-1)){
			getbatch(V,D1+N*Mt*(t+Tp),Omega1+N*Mt*(t+Tp),last_left1,last_right1,N,yichu1,R);	
			lsq(last_left1,last_right1,U+R*Mt*(t+Tp),N,yichu1,R);
			}else{
			getbatch(V,D1+N*Mt*(t+Tp),Omega1+N*Mt*(t+Tp),left1,right1,N,Mt,R);
			lsq(left1,right1,U+R*Mt*(t+Tp),N,Mt,R);
			}
		}

		cudaDeviceSynchronize();
		Matrixtranpose(U,M,R);
	}
		delete[] left;left=nullptr;
		delete[] right;right=nullptr;
		delete[] left1;left1=nullptr;
		delete[] right1;right1=nullptr;
		delete[] last_left;last_left=nullptr;
		delete[] last_right;last_right=nullptr;
		delete[] last_left1;last_left1=nullptr;
		delete[] last_right1;last_right1=nullptr;

	}else{

		dt *left = new dt[R*R*N]();
		dt *right = new dt[R*1*N]();
		dt *left1 = new dt[R*R*M]();
		dt *right1 = new dt[R*1*M]();

		for(int i = 0;i<10;i++){
			getbatch(U,D,Omega,left,right,M,N,R);
			lsq(left,right,V,M,N,R);
			Matrixtranpose(V,N,R);
	
			getbatch(V,D1,Omega1,left1,right1,N,M,R);
			lsq(left1,right1,U,N,M,R);
			Matrixtranpose(U,M,R);
		}

		delete[] left;left=nullptr;
		delete[] right;right=nullptr;
		delete[] left1;left1=nullptr;
		delete[] right1;right1=nullptr;
	}

	mproduct(U,V,result,M,R,N);
	double sh = 0.0;
	double xia = 0.0;
	for(int i = 0;i<M*N;i++){
		sh+=(result[i]-D[i])*(result[i]-D[i]);
		xia+=(D[i]*D[i]);
	}

	cout<<sqrt(sh)/sqrt(xia)<<endl;
	delete[] result;result=nullptr;
	// we need to U*V

	delete[] Omega1;Omega1=nullptr;
	delete[] Omega;Omega=nullptr;
	delete[] D1;D1=nullptr;
	delete[] V;V=nullptr;
	delete[] U;U=nullptr;
	delete[] D;D=nullptr;
    end = omp_get_wtime();
    double time = end-start;
    cout<<time<<endl;

	ofstream outfile("largemul.txt",ios::app);
	outfile<<M<<"*"<<N<<" "<<R<<" ";
	outfile<<time<<"s"<<" ";
	outfile<<sqrt(sh)/sqrt(xia)<<endl;
	outfile.close();
}
	return 0;
}

