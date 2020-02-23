/*************************************************************************
	> File Name: test.cpp
	> Author: hanlu
	> Mail: hanlu@shu.edu.cn 
	> Created Time: 2018年12月25日 星期二 12时31分14秒
 ************************************************************************/


#include "head.h"

int main(int argc,char *argv[]){

	int M = 4;
	int N = 3;
	int R = 2;
//	srand((unsigned)time(NULL));
	dt A[M*R] = {4,1,1,3,1,4,4,3};
	dt B[M*R] = {3,4,1,1,3,4};
	dt *D = new dt[M*N]();
	matrixProduct(A,B,D,M,R,N); // D is the origin data
	cout<<"origin is"<<endl;
	printTensor(D,1,M*N,1);
//	cout<<"origin data"<<endl;
//	printTensor(D,1,M*N,1);   // we view it store as col
	dt *D1 = new dt[M*N]();
	for(int i = 0;i<M;i++){
		for(int j = 0;j<N;j++){
			D1[i*N+j] = D[j*M+i];
		}
	}
	printTensor(D1,1,M*N,1);

	dt Omega[M*N]={1,0,1,1,0,1,1,0,1,1,0,1};
	cout<<"sample is"<<endl;
	printTensor(Omega,1,M*N,1);
	dt *Omega1 = new dt[M*N]();
	for(int i = 0;i<M;i++){
		for(int j = 0;j<N;j++){
			Omega1[i*N+j] = Omega[j*M+i];
		}
	}
	printTensor(Omega1,1,M*N,1);

	dt U[M*R] = {3,6,2,6,2,6,1,8};
	printTensor(U,1,M*R,1);

	dt *V = new dt[R*N]();
	dt *left = new dt[R*R*N]();
	dt *right = new dt[R*1*N]();
	dt *left1 = new dt[R*R*M]();
	dt *right1 = new dt[R*1*M]();
	for(int i = 0;i<1;i++){
		getbatch(U,D,Omega,left,right,M,N,R);
		cout<<"left is "<<endl;
		printTensor(left,1,R*R,N);
		cout<<"right is "<<endl;
		printTensor(right,1,R,N);
		lsq(left,right,V,M,N,R);
		printTensor(V,1,R*N,1);
	
		getbatch(V,D1,Omega1,left1,right1,N,M,R);
		lsq(left1,right1,U,N,M,R);
		printTensor(U,1,R*M,1);
		
	}
	dt *result = new dt[M*N]();

	mproduct(U,V,result,M,R,N);
//	printTensor(result,1,M*N,1);
	double sh = 0.0;
	double xia = 0.0;
	for(int i = 0;i<M*N;i++){
		sh+=(result[i]-D[i])*(result[i]-D[i]);
		xia+=(D[i]*D[i]);
	}
	cout<<sqrt(sh)/sqrt(xia)<<endl;

	delete[] result;result=nullptr;

	delete[] left;left=nullptr;
	delete[] Omega1;Omega1=nullptr;
	delete[] D;D=nullptr;
	delete[] right;right=nullptr;
	delete[] left1;left=nullptr;
	delete[] D1;D1=nullptr;
	delete[] right1;right=nullptr;

	return 0;

}
