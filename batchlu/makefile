#cc = nvcc
des = luhan
#source = sinsol.cpp opera.cu newkernel.cu batchlu.cu
#source = test.cpp opera.cu kernel.cu altmin.cu
#source = realdata.cpp opera.cu newkernel.cu optlsq.cu
#source = solverlu.cpp opera.cu newkernel.cu batchlu.cu
source = multicore.cu opera.cu newkernel.cu batchlu.cu
#source =lu-2019-7-1.cpp opera.cu kernel.cu batchlu.cu batchedtprod.cu
link = -lcufft  -lcublas -lcusolver -lmagma 
cc = /usr/local/cuda-10.1/bin/nvcc
include = -I /usr/local/magma/include
lib = -L /usr/local/magma/lib
flags = -DADD_ -std=c++11 

$(des):$(source)
	$(cc) -o $(des) $(source) $(include) $(lib) $(link) $(flags)

clean:
	rm -rf $(des)
