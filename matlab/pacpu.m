
close all;
clear all;
rng(234923); 
for iter =1:2 
%array=[512,128,256,512,1024,2048,4096,8192];
array=[2000,3000];
MM = array(iter);
NN = MM;
%RR =ceil(0.01*MM);
RR = 0.01*MM;
D = rand(MM,RR)*rand(RR,NN);
rho=0.5;
Omega = rand(MM,NN)<=rho; % support of observation
Omega1 =Omega';

D_omega = Omega.*D; % measurements are made
D_omega1= D_omega';
parpool('local',20);

tic
for t=1:10  %循环次数
    if t==1
        U = rand(MM,RR);
%         temp=D_omega.*P;
      %   [U,S1,V1]=svds(D_omega,RR);
    end

   parfor j=1:NN %分别算V的每一列
        for i=1:RR
            x1(:,i,j)=Omega(:,j).*U(:,i);%将采样矩阵K的第j列跟U0对应行的每个元素相乘
        end
    end

    parfor j = 1:NN
        v(:,j)=pinv(x1(:,:,j))*D_omega(:,j); %算v的第j列
    end

   % v=v+normrnd(0,sigma^2*T);
    V=v';
    parfor j=1:MM %分别算U的每一行
        for i=1:RR
            x2(:,i,j)=Omega1(:,j).*V(:,i);
        end
    end

    parfor j= 1:MM
       u(:,j)=pinv(x2(:,:,j))*D_omega1(:,j); %算A'的第j列
    end

    U=u';
    
   % display(p(t));
    %display(p2(t));
end
    V=V';
   % U=U+normrnd(0,sigma^2*T);
    M0=U*V;  
    time = toc;

    fid = fopen('cpu.txt','a');
    fprintf(fid,'%d*%d %d %fs\n',MM,NN,RR,time);
    fclose(fid);
    p=norm(D-M0,'fro')/norm(D(:));
    display(p);
    delete(gcp);
    clear all;
end
