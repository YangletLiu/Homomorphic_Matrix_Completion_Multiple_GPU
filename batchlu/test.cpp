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
//	cout<<"origin data"<<endl;
//	printTensor(D,1,M*N,1);   // we view it store as col

	dt Omega[M*N]={1,0,1,1,0,1,1,0,1,1,0,1};
	dt U[M*R] = {3,6,2,6,2,6,1,8};

	dt *V = new dt[R*N]();

	dt *left = new dt[R*R*N]();
	dt *right = new dt[R*1*N]();
	cout<<"origin data"<<endl;
	printTensor(D,1,M*N,1);
	cout<<"sample data"<<endl;
	printTensor(Omega,1,M*N,1);
	cout<<"U"<<endl;
	printTensor(U,1,M*R,1);

	getbatch(U,D,Omega,left,right,M,N,R);
	cout<<"left"<<endl;
	printTensor(left,1,R*R,N);
	cout<<"right"<<endl;
	printTensor(right,1,R*1,N);

//	newmode(U,D,Omega,left,right,M,N,R);
	lsq(left,right,V,M,N,R);
	printTensor(V,1,R*N,1);

	delete[] left;left=nullptr;
	delete[] D;D=nullptr;
	delete[] right;right=nullptr;
	delete[] V;V=nullptr;
	return 0;

}
