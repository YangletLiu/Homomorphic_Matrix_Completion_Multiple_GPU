
function L = max12norm(A,B)

[m,~] = size(A);
DD = B.*A;
max=0;
for i=1:m
	buf = norm(DD(i,:));
	if max<buf
		max=buf;
	end
end

L=max;
