/*************************************************************************
	> File Name: trust.cpp
	> Author: hanlu
	> Mail: hanlu@shu.edu.cn 
	> Created Time: 2018年12月10日 星期一 21时13分39秒
 ************************************************************************/

#include "head.h"

int main(int argc,char *argv[]){
	//int luhan[10] = {1024,300,128,256,512,1024,2048,4096,8192,100};
	int luhan[4] = {256,256,8182,16384};
	for(int hh =0;hh<4;hh++){
	int M = luhan[hh];
	int N = M;
	int R;
	int T;
	cout<<M<<endl;	
	clock_t t1,t2;
	t1=clock();
	if(M>=N){
	 R = int(M*0.01)+1;
	}else{
	 R = int(N*0.01)+1;
	}
	double real = (M/4500.0)*(N/4500.0)*(R/45.0);
//	double real = (M/2200.0)*(N/2200.0)*(R/220.0);
	int rr = int(real);
	cout<<real<<endl;
	if((real-rr)>0){
		T = rr+1;
	}else{
		T = rr;
	}
	cout<<T<<endl;
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
	dt *PB = new dt[M*K]();   //public data size of M*R store as column
	for(int i = 0;i<M*K;i++){
		PB[i] = D[i];
	}
	
	dt *MM = new dt[M*N]();  //MM is encpypted matrix
	dt *MM1 = new dt[M*N]();  //MM1 is its transpose
	Enc(D,Omega,PB,MM,MM1,M,N);

	if(T>1){

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
	//	cout<<Nt<<"  "<<Mt<<endl;
		int yichu,yichu1;
		yichu = N-(T-1)*Nt;
		yichu1 = M-(T-1)*Mt;
	//	cout<<yichu<<"  "<<yichu1<<endl;

		
		dt *last_left = new dt[R*R*yichu]();
		dt *last_right = new dt[R*1*yichu]();
		dt *left = new dt[R*R*Nt]();
		dt *right = new dt[R*1*Nt]();
		dt *left1 = new dt[R*R*Mt]();
		dt *right1 = new dt[R*1*Mt]();
		dt *last_left1 = new dt[R*R*yichu1]();
		dt *last_right1 = new dt[R*1*yichu1]();

	for(int i = 0;i<10;i++){
		
		for(int t = 0;t<T;t++){
			if(t==(T-1)){
			getbatch(U,MM+M*Nt*t,Omega+M*Nt*t,last_left,last_right,M,yichu,R);	
			lsq(last_left,last_right,V+R*Nt*t,M,yichu,R);
	//			printTensor(last_left,1,R*R,yichu);
	//			printTensor(last_right,1,R,yichu);
	//			printTensor(V+R*Nt*t,1,R,yichu);

			}else{
			getbatch(U,MM+M*Nt*t,Omega+M*Nt*t,left,right,M,Nt,R);	
			lsq(left,right,V+R*Nt*t,M,Nt,R);
			}
	//	cout<<"left is"<<endl;
	//	printTensor(left,1,R*R,Nt);
	//	cout<<"right is"<<endl;
	//	printTensor(right,1,R,Nt);
		//	cout<<"V part"<<endl;
		//	printTensor(V+Nt*t,1,R,Nt);

		}
		//cout<<"luhan"<<endl;
			Matrixtranpose(V,N,R);
//			printTensor(V,1,R*N,1);

		for(int t = 0;t<T;t++){
			if(t==(T-1)){
			getbatch(V,MM1+N*Mt*t,Omega1+N*Mt*t,last_left1,last_right1,N,yichu1,R);	
			lsq(last_left1,last_right1,U+R*Mt*t,N,yichu1,R);
			}else{
			getbatch(V,MM1+N*Mt*t,Omega1+N*Mt*t,left1,right1,N,Mt,R);
			lsq(left1,right1,U+R*Mt*t,N,Mt,R);
			}
		
		}
			Matrixtranpose(U,M,R);
//			printTensor(U,1,R*M,1);
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
			getbatch(U,MM,Omega,left,right,M,N,R);
			lsq(left,right,V,M,N,R);
			Matrixtranpose(V,N,R);
	
			getbatch(V,MM1,Omega1,left1,right1,N,M,R);
			lsq(left1,right1,U,N,M,R);
			Matrixtranpose(U,M,R);
		}

		delete[] left;left=nullptr;
		delete[] right;right=nullptr;
		delete[] left1;left1=nullptr;
		delete[] right1;right1=nullptr;
	}
	dt *result1 = new dt[M*N]();
	mproduct(U,V,result1,M,R,N);
	Dec(result1,result,PB,M,N); // decrypt result1 to result
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
	delete[] result1;result1=nullptr;
	delete[] MM;MM=nullptr;
	delete[] MM1;MM1=nullptr;
	t2=clock();
	ofstream outfile("speedup.txt",ios::app);
	outfile<<M<<"*"<<N<<" "<<R<<" ";
	outfile<<(double)(t2-t1)/CLOCKS_PER_SEC<<"s"<<" ";
	outfile<<sqrt(sh)/sqrt(xia)<<endl;
	outfile.close();
}
	return 0;
}

