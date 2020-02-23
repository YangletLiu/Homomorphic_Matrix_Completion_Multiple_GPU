/*************************************************************************
	> File Name: trust.cpp
	> Author: hanlu
	> Mail: hanlu@shu.edu.cn 
	> Created Time: 2018年12月10日 星期一 21时13分39秒
 ************************************************************************/

#include "head.h"

int main(int argc,char *argv[]){

	int M =1000;
	int N = M;
	int R = int(0.01*M);
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

	dt *Omega1 = new dt[M*N]();
	for(int i = 0;i<M;i++){
		for(int j = 0;j<N;j++){
			Omega1[i*N+j] = Omega[j*M+i];
		}
	}
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
	clock_t t1,t2;
	t1 = clock();

	for(int i = 0;i<10;i++){
		base(U,MM,Omega,V,M,N,R);
		base(V,MM1,Omega1,V,N,M,R);
	}
	dt *result = new dt[M*N]();
	dt *result1 = new dt[M*N]();
	mproduct(U,V,result1,M,R,N);
	t2 = clock();
	cout<<M<<"*"<<N<<" "<<(double)(t2-t1)/CLOCKS_PER_SEC<<"s"<<endl;
	Dec(result1,result,PB,M,N); // decrypt result1 to result

	delete[] Omega1;Omega1=nullptr;
	delete[] Omega;Omega=nullptr;
	delete[] D1;D1=nullptr;
	delete[] U;U=nullptr;
	delete[] V;V=nullptr;
	
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
/*	ofstream outfile("runtime.txt",ios::app);
	outfile<<M<<"*"<<N<<" "<<R<<" ";
	//	t3 = clock();
	outfile<<(double)(t2-t1)/CLOCKS_PER_SEC<<"s"<<"  ";

	cout<<M<<"*"<<N<<" "<<(double)(t2-t1)/CLOCKS_PER_SEC<<"s"<<endl;
	outfile<< error<<endl;
	outfile.close();

}*/
	return 0;
}

