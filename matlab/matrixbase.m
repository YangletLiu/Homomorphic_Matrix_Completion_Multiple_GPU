
clear all
for iter = 1:10
%	array=[700,1024,1500,2048,3000,3500,4096,4500,8192,16384];
	array=[700,500,1000,1500,2000,2500,3000,3500,4000,4500];
%	array=[700,800,500,16384];
	MM = array(iter);
	NN = MM;
	A=rand(MM,NN);
	tic
	for i=1:10
		res=A'*A;
	end
	time=toc;
	tt=time/10;
    	fid = fopen('mbase.txt','a');
    	fprintf(fid,'%d %d %gs\n',MM,NN, tt);
    	fclose(fid);
end

