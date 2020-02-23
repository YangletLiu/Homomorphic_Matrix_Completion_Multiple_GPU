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

//int luhan[11]={600,500,1024,1500,2048,3000,3500,4096,4500,8192,16384};
//int luhan[11]={1000,300,700,500,1000};
int luhan[19] = {1000,500,1000,1500,2000,3000,2500,3500,4000,4500,500,1000,1500,2000,2500,3000,3500,4000,4500};
for(int hh=0;hh<19;hh++){
	srand((unsigned)time(NULL));
	int m = luhan[hh];
	int n = 1;
	int r = int(0.01*m)+1;
	
	dt *b = new dt[m*n]();
	for(int i = 0;i<m*n;i++){
		b[i] = rand()*0.1/(RAND_MAX*0.1);
	}

	dt *x = new dt[r*n]();

        dt *U = new dt[m*r]();
	for(int i = 0;i<m*r;i++){
		U[i] = rand()*0.1/(RAND_MAX*0.1);
	}
	dt *tt = new dt[r]();
	clock_t start,end,time;
	start = clock();
for(int j = 0;j<10;j++){

	base(U,tt,b,m,r,n);
}
	delete[] tt; tt=nullptr;
//}
	end = clock();
	time = (end-start)/10;
	ofstream outfile("base.txt",ios::app);
	outfile<<m<<"*"<<r<<" ";
	outfile<<(double)(time)/CLOCKS_PER_SEC<<"s"<<endl;
	outfile.close();
	
	delete[] U; U = nullptr;
	delete[] b; b = nullptr;
	delete[] x; x = nullptr;
}
/*
for(int ss = 0;ss<19;ss++){
	srand((unsigned)time(NULL));
	int M = luhan[ss];
	dt *A = new dt[M*M]();
	for(int i = 0;i<M*M;i++){
		A[i] = rand()*0.1/(RAND_MAX*0.1);
	}
	dt *res = new dt[M*M]();
	clock_t t1,t2,tt;
	t1 = clock();
	for(int j = 0;j<10;j++){
		basematrixpro(A,res,M);
	}
	t2 = clock();
	tt = (t2-t1)/10;
	ofstream outfile("base.txt",ios::app);
	outfile<<M<<"*"<<M<<" ";
	outfile<<(double)(tt)/CLOCKS_PER_SEC<<"s"<<endl;
	outfile.close();
	delete[] A;A=nullptr;
	delete[] res;res=nullptr;
}
*/
	return 0;
}

