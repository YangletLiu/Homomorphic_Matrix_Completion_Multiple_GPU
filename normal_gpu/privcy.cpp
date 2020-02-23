/*************************************************************************
	> File Name: trust.cpp
	> Author: hanlu
	> Mail: hanlu@shu.edu.cn 
	> Created Time: 2018年12月10日 星期一 21时13分39秒
 ************************************************************************/

#include "head.h"

int main(int argc,char *argv[]){

	int m = 2000;
	int n = 2000;
	int r = 20;

	srand((unsigned)time(NULL));
	dt *A = new dt[m*r]();
	for(int i = 0;i<m*r;i++){
		A[i] = rand()*0.1/(RAND_MAX*0.1);
	}

	dt *B = new dt[r*n]();
	for(int i = 0;i<r*n;i++){
		B[i] = rand()*0.1/(RAND_MAX*0.1);
	}

	dt *D = new dt[m*n]();
	matrixProduct(A,B,D,m,r,n);
	delete[] A;A=nullptr;
	delete[] B;B=nullptr;
	
	dt *D1  = new dt[m*n]();

	for(int i = 0;i<m;i++){
		for(int j = 0;j<n;j++){
			D1[i*n+j] = D[j*m+i];
		}
	}

	dt *Omega = new dt[m*n]();
	for(int i = 0;i<m*n;i++){
		Omega[i] = rand()*0.1/(RAND_MAX*0.1);
		if(Omega[i]<=0.5){
			Omega[i] = 1;
		}else{
			Omega[i] = 0;
		}
	}

	dt *O1  = new dt[m*n];
	for(int i = 0;i<n;i++){
		for(int j = 0;j<m;j++){
			O1[j*n+i] = Omega[i*m+j];
		}
	}
	dt *miss = new dt[m*n]();
	for(int i=0;i<m*n;i++){
		miss[i] = D1[i]*O1[i];
	}

	for(int i = 0;i<5*n;i++){
		miss[i] = D1[i];
	}
	for(int i = 5;i<m;i++){
		for(int j = 0;j<n;j++){
			miss[i*n+j] =(0.5*miss[i*n+j]+0.1*miss[j]+0.1*miss[n+j]+0.1*miss[2*n+j]+0.1*miss[3*n+j]+0.1*miss[4*n+j])*O1[i*n+j];
		}
	}

	dt *D_omega=new dt[m*n]();
	for(int i = 0;i<n;i++){
		for(int j = 0;j<m;j++){
			D_omega[i*m+j] = miss[j*n+i]; 
		}
	}


	dt *U = new dt[m*r]();
	for(int i = 0;i<m*r;i++){
		U[i] = rand()*0.1/(RAND_MAX*0.1);
	}

	dt *V = new dt[r*n]();

	for(int i =0;i<20;i++){
		lsq(U,V,D_omega,Omega,m,r,n);
		lsq(V,U,miss,O1,n,r,m);

	dt *result = new dt[m*n]();
	dt *result1 = new dt[m*n]();
	mproduct(U,V,result,m,r,n);
	
	for(int i = 0;i<m;i++){
		for(int j = 0;j<n;j++){
			result1[i*n+j] = result[j*m+i];
		}
	}
	for(int i = 5;i<m;i++){
		for(int j = 0;j<n;j++){
			result1[i*n+j] =(result1[i*n+j]-0.1*miss[j]-0.1*miss[n+j]-0.1*miss[2*n+j]-0.1*miss[3*n+j]-0.1*miss[4*n+j])/0.5;
		}
	}
	for(int i = 0;i<m;i++){
		for(int j = 0;j<n;j++){
			result[j*m+i] = result1[i*n+j];
		}
	}
	
	double sh = 0.0;
	double xia = 0.0;
	for(int i = 0;i<m*n;i++){
		sh += ((result[i]-D[i])*(result[i]-D[i]));
		xia +=(D[i]*D[i]);
	}
	
	double error=0.0;
	error = sqrt(sh)/sqrt(xia);
	cout<<error<<endl;
	ofstream outfile("error.txt",ios::app);
	outfile<<i<<" "<<m<<"*"<<n<<" ";
	outfile<<error<<endl;
	outfile.close();

	delete[] result; result = nullptr;
	delete[] result1; result1 = nullptr;
	}

	delete[] D_omega; D_omega = nullptr;
	delete[] Omega; Omega = nullptr;
	delete[] U; U = nullptr;
	delete[] V; V = nullptr;
	delete[] D1; D1 = nullptr;
	delete[] D; D = nullptr;
	delete[] O1; O1 = nullptr;
	delete[] miss; miss = nullptr;

	return 0;
}

