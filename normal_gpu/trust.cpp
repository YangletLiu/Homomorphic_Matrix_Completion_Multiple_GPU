/*************************************************************************
	> File Name: trust.cpp
	> Author: hanlu
	> Mail: hanlu@shu.edu.cn 
	> Created Time: 2018年12月10日 星期一 21时13分39秒
 ************************************************************************/

#include "head.h"

int main(int argc,char *argv[]){
//	int m = atoi(argv[1]);	
//	int n = atoi(argv[2]);	
//	int rank = (m*0.1<n*0.1)?m*0.1:n*0.1;

//int luhan[9]={500,1000,1500,2000,2500,3000,3500,4000,4500};
//int luhan[3]={128,256,16384};
//int luhan[18]={256,128,500,800,1000,1200,1500,1800,2000,2100,2200,128,256,512,1024,2048,4096,8192};
//for(int hh=0;hh<3;hh++){
//	int m = luhan[hh];
	int m = 4500;
	int n = m;
	int r = int(0.01*m);
	
	srand((unsigned)time(NULL));
	dt *A = new dt[m*r]();
	for(int i = 0;i<m*r;i++){
		A[i] = rand()*0.1/(RAND_MAX*0.1);
	}
//	printTensor(A,m,r,1);
	dt *B = new dt[r*n]();
	for(int i = 0;i<r*n;i++){
		B[i] = rand()*0.1/(RAND_MAX*0.1);
	}
//	printTensor(B,r,n,1);
	dt *D = new dt[m*n]();
	matrixProduct(A,B,D,m,r,n);
	delete[] A;A=nullptr;
	delete[] B;B=nullptr;
//	printTensor(D,n,m,1);    
	//we get the transpose result
	// if we transfer to GPU we view it as m*n;

//	dt D[m*n]={10,3,12,4,3,6,7,45,67,34,4,1};
	dt *D1  = new dt[m*n]();
	for(int i = 0;i<n;i++){
		for(int j = 0;j<m;j++){
			D1[j*n+i] = D[i*m+j];
		}
	}

//	dt Omega[m*n]={1,0,1,1,0,1,1,0,1,1,0,1};
	dt *Omega = new dt[m*n]();
	for(int i = 0;i<m*n;i++){
		Omega[i] = rand()*0.1/(RAND_MAX*0.1);
		if(Omega[i]<=0.5){
			Omega[i] = 1;
		}else{
			Omega[i] = 0;
		}
	}
//	printTensor(Omega,m,n,1);

	dt *O1  = new dt[m*n];
	for(int i = 0;i<n;i++){
		for(int j = 0;j<m;j++){
			O1[j*n+i] = Omega[i*m+j];
		}
	}
//	printTensor(O1,1,m*n,1);

	dt *D_omega=new dt[m*n]();
	for(int i = 0;i<m*n;i++){
		D_omega[i] = D[i]*Omega[i];
	}

	dt *D1_O1=new dt[m*n]();
	for(int i = 0;i<m*n;i++){
		D1_O1[i] = D1[i]*O1[i];
	}
//	printTensor(D1_O1,1,m*n,1);

//	printTensor(D_omega,n,m,1);

//	dt U[m*r] = {7,4,12,5,1,5,3,4};
    dt *U = new dt[m*r]();
	for(int i = 0;i<m*r;i++){
		U[i] = rand()*0.1/(RAND_MAX*0.1);
	}
//	printTensor(U,1,m*r,1);

	dt *V = new dt[r*n]();
//	test(U,Omega,lulu,m,r,n);	

	clock_t start,end;
	start = clock();

	for(int i =0;i<10;i++){
		lsq(U,V,D_omega,Omega,m,r,n);
		lsq(V,U,D1_O1,O1,n,r,m);
//	printTensor(U,1,m*r,1);
//	printTensor(V,1,n*r,1);
	}


// U is m*r V is r*n 
	dt *result = new dt[m*n]();
	mproduct(U,V,result,m,r,n);
	end = clock();
//	ofstream outfile("unopten.txt",ios::app);
//	outfile<<m<<"*"<<n<<" "<<r<<" ";
	//	t3 = clock();
//	outfile<<(double)(end-start)/CLOCKS_PER_SEC<<"s"<<"  ";
	
	for(int i = 0;i<m*n;i++){
		if(isnan(result[i])||isnan(D[i])){
			result[i] = 0;
			D[i] = 0;
		}
	}
//	printTensor(result,1,m*n,1);
//	printTensor(D,1,m*n,1);

	double sh = 0.0;
	double xia = 0.0;
	for(int i = 0;i<m*n;i++){
		sh += (result[i]-D[i])*(result[i]-D[i]);
		xia +=(D[i]*D[i]);
	}
	
	double error=0.0;
	error = sqrt(sh)/sqrt(xia);
	cout<<error<<endl;
//	outfile<<error<<endl;
//	outfile.close();
	

	delete[] D_omega; D_omega = nullptr;
	delete[] Omega; Omega = nullptr;
	delete[] U; U = nullptr;
	delete[] V; V = nullptr;
	delete[] D1; D1 = nullptr;
	delete[] D1_O1; D1_O1 = nullptr;
	delete[] D; D = nullptr;
	delete[] result; result = nullptr;
	delete[] O1; O1 = nullptr;
//}
	return 0;
}

