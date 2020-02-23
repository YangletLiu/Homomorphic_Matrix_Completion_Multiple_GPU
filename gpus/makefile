des = luhan

source = solverlu.cpp opera.cu newkernel.cu altmin.cu

#source = gpus_mul.cu opera.cu newkernel.cu batchlu.cu

link = -lcufft  -lcublas -lcusolver -lmagma 
cc = /usr/local/cuda-10.1/bin/nvcc
include = -I /usr/local/magma/include
lib = -L /usr/local/magma/lib
flags =  -DADD_ -std=c++11 

$(des):$(source)
	$(cc) -lgomp -o $(des) $(source) $(include) $(lib) $(link) $(flags)

clean:
	rm -rf $(des)
