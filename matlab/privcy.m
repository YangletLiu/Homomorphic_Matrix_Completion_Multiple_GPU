
clear

rng(234923);
MM = 2000;
NN = 2000;
RR = 20;
D = rand(MM,RR)*rand(RR,NN);
rho=0.5;
%for ii = 1:9
Omega = rand(MM,NN)<=rho; % support of observati
D_omega = D.*Omega; 
t1=clock;
miss = rand(5,NN);
for i =1:MM
    D_omega(i,:)= (0.5*D_omega(i,:)+0.1*miss(1,:)+0.1*miss(2,:)+0.1*miss(3,:)+0.1*miss(4,:)+0.1*miss(5,:)).*Omega(i,:);
end
D_omega1= D_omega';
Omega1 =Omega';

t2 = clock;

for t=1:10 %循环次数
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
end
    M0=U*V';  
    t3 = clock;
    % we will 解密
    
    for i= 1:MM
        M0(i,:)=(M0(i,:)-(0.1*miss(1,:)+0.1*miss(2,:)+0.1*miss(3,:)+0.1*miss(4,:)+0.1*miss(5,:)))/0.5;
    end
    p=norm(D-M0,'fro')/norm(D(:));
    display(p);
    t4=clock;
    display(etime(t2,t1));
    display(etime(t3,t2));
    display(etime(t4,t3));
  




