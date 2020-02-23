cc = /usr/local/cuda-10.1/bin/nvcc
des = luhan
#source = hanlu.cpp opera.cu kernel.cu altmin1.cu 
source = hanlu.cpp opera.cu kernel.cu batchlu.cu 
#source = sinsol.cpp opera.cu newkernel.cu batchlu.cu 
link = -lcublas -lcusolver -lmagma 
include = -I /usr/local/magma/include
lib = -L /usr/local/magma/lib
flags = -DADD_ -std=c++11 

$(des):$(source)
	$(cc) -o $(des) $(source) $(include) $(lib) $(link) $(flags)


clean:
	rm -rf $(des)
