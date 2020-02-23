/*************************************************************************
	> File Name: trust.cpp
	> Author: hanlu
	> Mail: hanlu@shu.edu.cn 
	> Created Time: 2018年12月10日 星期一 21时13分39秒
 ************************************************************************/

#include "head.h"

int main(int argc,char *argv[]){
	int m = 4;
	int n = 3;
	int r = 2;

	dt D[m*n]={10,3,12,4,3,6,7,45,67,34,4,1};
	/*
	 * the origin maxtrix
	 *	10  3   67
	 *	3   6   34 
	 *	12  7   4
	 *	4   45  1    M*N 4*3
	 * */ 
	dt *D1  = new dt[m*n]();
	for(int i = 0;i<n;i++){
		for(int j = 0;j<m;j++){
			D1[j*n+i] = D[i*m+j];
		}
	}     // origin matrix transpose

	dt Omega[m*n]={1,0,1,1,0,1,1,0,1,1,0,1};
	/*
	 *  sample 
	 *	1  0  1
	 *	0  1  1
	 *	1  1  0
	 *  1  0  1     M*N 4*3
	 *
	 * */
	dt *O1  = new dt[m*n]();
	for(int i = 0;i<n;i++){
		for(int j = 0;j<m;j++){
			O1[j*n+i] = Omega[i*m+j];
		}
	}     //sample transpose

	dt *D_omega=new dt[m*n]();
	for(int i = 0;i<m*n;i++){
		D_omega[i] = D[i]*Omega[i];
	}
	/*  the loss matrix
	 *  
	 *  10  0  0
	 *	0   6  34
	 *	12  7  0
	 *  4   0  1   M*N 4*3
	 * */
	
	dt *D1_O1=new dt[m*n]();
	for(int i = 0;i<m*n;i++){
		D1_O1[i] = D1[i]*O1[i];
	}

	dt U[m*r] = {7,4,12,5,1,5,3,4};
//	dt U[m*r] = {1.28193,0.354167,-14.5556,-0.721601,0.916667,18.4444};
	/*
	 * inital U
	 *	7   1
	 *	4   5
	 *  12  3
	 *  5   4    M*R 4*2
	 * */

	dt *V = new dt[r*n]();
//	dt *U = new dt[r*m]();
	
	for(int i = 0;i<3;i++){
		lsq(U,V,D_omega,Omega,m,r,n);
		lsq(V,U,D1_O1,O1,n,r,m);
		printTensor(V,1,n*r,1);
		printTensor(U,1,m*r,1);
	}

	dt *M = new dt[m*n]();
	mproduct(U,V,M,m,r,n);
	printTensor(M,1,m*n,1);
	delete[] M;M=nullptr;
	printTensor(D,1,m*n,1);

	delete[] D_omega; D_omega = nullptr;
//	delete[] U; U = nullptr;
	delete[] V; V = nullptr;
	delete[] D1; D1 = nullptr;
	delete[] D1_O1; D1_O1 = nullptr;
	delete[] O1; O1 = nullptr;

	return 0;
}

