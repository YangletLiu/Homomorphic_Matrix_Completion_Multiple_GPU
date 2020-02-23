/*************************************************************************
	> File Name: trust.cpp
	> Author: hanlu
	> Mail: hanlu@shu.edu.cn 
	> Created Time: 2018年12月10日 星期一 21时13分39秒
 ************************************************************************/

#include "head.h"

int main(int argc,char *argv[]){
	int M = 4;
	int N = 3;
	int R = 2;
	int T = int((M*N*R+base-1)/base);
	cout<<T<<endl;
	srand((unsigned)time(NULL));
	dt A[M*R] = {4,1,1,3,1,4,4,3};
	dt B[M*R] = {3,4,1,1,3,4};

	dt *D = new dt[M*N]();
	matrixProduct(A,B,D,M,R,N); // D is the origin data
	cout<<"origin is"<<endl;
	printTensor(D,1,M*N,1);

	// get transpose of D  N*M
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

	dt *V = new dt[R*N]();
	dt *result = new dt[M*N]();
	
	clock_t t1,t2;

	if(T>1){

		int Nt,Mt;
		if (N%2!=0){
		 Nt = int(N/T)+1;
		}else{
			 Nt = N/T;
		}
		if (M%2!=0){
		 Mt = int(M/T)+1;
		}else{
			 Mt = M/T;
		}
		cout<<Nt<<"  "<<Mt<<endl;
		int yichu,yichu1;
		yichu = N-(T-1)*Nt;
		yichu1 = M-(T-1)*Mt;
		cout<<yichu<<"  "<<yichu1<<endl;

		
		dt *last_left = new dt[R*R*yichu]();
		dt *last_right = new dt[R*1*yichu]();
		dt *left = new dt[R*R*Nt]();
		dt *right = new dt[R*1*Nt]();
		dt *left1 = new dt[R*R*Mt]();
		dt *right1 = new dt[R*1*Mt]();
		dt *last_left1 = new dt[R*R*yichu1]();
		dt *last_right1 = new dt[R*1*yichu1]();

	t1 = clock();
	for(int i = 0;i<1;i++){
		
		for(int t = 0;t<T;t++){
			if(t==(T-1)){
			getbatch(U,D+M*Nt*t,Omega+M*Nt*t,last_left,last_right,M,yichu,R);	
			lsq(last_left,last_right,V+R*Nt*t,M,yichu,R);
	//			printTensor(last_left,1,R*R,yichu);
	//			printTensor(last_right,1,R,yichu);
	//			printTensor(V+R*Nt*t,1,R,yichu);

			}else{
			getbatch(U,D+M*Nt*t,Omega+M*Nt*t,left,right,M,Nt,R);	
			lsq(left,right,V+R*Nt*t,M,Nt,R);
			}
	//	cout<<"left is"<<endl;
	//	printTensor(left,1,R*R,Nt);
	//	cout<<"right is"<<endl;
	//	printTensor(right,1,R,Nt);
		//	cout<<"V part"<<endl;
		//	printTensor(V+Nt*t,1,R,Nt);

		}
		cout<<"luhan"<<endl;
			Matrixtranpose(V,N,R);
			printTensor(V,1,R*N,1);

		for(int t = 0;t<T;t++){
			if(t==(T-1)){
			getbatch(V,D1+N*Mt*t,Omega1+N*Mt*t,last_left1,last_right1,N,yichu1,R);	
			lsq(last_left1,last_right1,U+R*Mt*t,N,yichu1,R);
			}else{
			getbatch(V,D1+N*Mt*t,Omega1+N*Mt*t,left1,right1,N,Mt,R);
			lsq(left1,right1,U+R*Mt*t,N,Mt,R);
			}
		
		}
			Matrixtranpose(U,M,R);
			printTensor(U,1,R*M,1);
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

		t1 = clock();
		for(int i = 0;i<1;i++){

			getbatch(U,D,Omega,left,right,M,N,R);
			lsq(left,right,V,M,N,R);
			Matrixtranpose(V,N,R);
			printTensor(V,1,R*N,1);
	
			getbatch(V,D1,Omega1,left1,right1,N,M,R);
			lsq(left1,right1,U,N,M,R);
			Matrixtranpose(U,M,R);
			printTensor(U,1,R*M,1);
		}

		delete[] left;left=nullptr;
		delete[] right;right=nullptr;
		delete[] left1;left1=nullptr;
		delete[] right1;right1=nullptr;
	}
	mproduct(U,V,result,M,R,N);
	t2=clock();
//	printTensor(result,1,M*N,1);


	double sh = 0.0;
	double xia = 0.0;
	for(int i = 0;i<M*N;i++){
		sh+=(result[i]-D[i])*(result[i]-D[i]);
		xia+=(D[i]*D[i]);
	}
/*	ofstream outfile("speedup.txt",ios::app);
	outfile<<M<<"*"<<N<<" ";
	//	t3 = clock();
	outfile<<(double)(t2-t1)/CLOCKS_PER_SEC<<"s"<<endl;
	outfile.close();
*/
	cout<<sqrt(sh)/sqrt(xia)<<endl;
	delete[] result;result=nullptr;
	// we need to U*V

	delete[] Omega1;Omega1=nullptr;
	delete[] D1;D1=nullptr;
	delete[] D;D=nullptr;
	delete[] V;V=nullptr;
	return 0;
}

