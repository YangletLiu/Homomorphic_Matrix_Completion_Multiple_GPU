
close all;
clear all;

rng(234923); 
for iter =1:8 
array=[512,128,256,512,1024,2048,4096,8192];
MM = array(iter);
NN = MM;
RR =ceil(0.01*MM);
D = rand(MM,RR)*rand(RR,NN);
rho=0.5;
Omega = rand(MM,NN)<=rho; % support of observation
Omega1 =Omega';

D_omega = Omega.*D; % measurements are made
for i=1:5
	PB(:,i) = D(:,i);
end
for i = 1:NN
	D_omega(:,i) =(0.5*D_omega(:,i)+0.1*PB(:,1)+0.1*PB(:,2)+0.1*PB(:,3)+0.1*PB(:,4)+0.1*PB(:,5)).*Omega(:,i);
end
D_omega1= D_omega';
tic

for t=1:10  %循环次数
    if t==1
        U = rand(MM,RR);
%         temp=D_omega.*P;
      %   [U,S1,V1]=svds(D_omega,RR);
    end
    for j=1:NN %分别算V的每一列
        y=D_omega(:,j);        
        for i=1:RR
            x1(:,i)=Omega(:,j).*U(:,i);%将采样矩阵K的第j列跟U0对应行的每个元素相乘
        end
        v(:,j)=pinv(x1)*y; %算v的第j列
    end
   % v=v+normrnd(0,sigma^2*T);
    V=v';
    for j=1:MM %分别算U的每一行
        y=D_omega1(:,j);
        for i=1:RR
            x2(:,i)=Omega1(:,j).*V(:,i);
        end
        u(:,j)=pinv(x2)*y; %算A'的第j列
    end
    U=u';
    
   % display(p(t));
    %display(p2(t));
end

    V=V';
   % U=U+normrnd(0,sigma^2*T);
    M0=U*V;  
    time = toc;
for i = 1:NN
	M0(:,i) =(M0(:,i)-0.1*PB(:,1)-0.1*PB(:,2)-0.1*PB(:,3)-0.1*PB(:,4)-0.1*PB(:,5))/0.5;
end

    fid = fopen('iterten.txt','a');
    fprintf(fid,'%d*%d %d %fs\n',MM,NN,RR,time);
    fclose(fid);
    p=norm(D-M0,'fro')/norm(D(:));
    display(p);
    clear all;
end
