clear all

rng(234923);
for iter =1:3
   % array=[600,500,1024,1500,2048,3000,3500,4096,4500,8192,16384];
    array=[600,1000,1000];
    MM = array(iter);
    NN = MM;
    RR = ceil(0.01*MM);
    b = rand(MM,1);
    U = rand(MM,RR);
    x = zeros(RR,1);
    tic
    for j = 1:10
    	x=pinv(U)*b;
    end

    tt=toc;
    time=tt/10;
    fid = fopen('tbase.txt','a');
    fprintf(fid,'%d %gs\n',MM,time);
    fclose(fid);
    clear all;

end   
