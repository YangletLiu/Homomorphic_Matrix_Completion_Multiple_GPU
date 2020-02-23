cc = nvcc
des = luhan
source = trust.cpp opera.cu kernel.cu altmin.cu
#source = privcy.cpp opera.cu kernel.cu altmin.cu
#source = platcompare.cpp opera.cu kernel.cu altmin.cu
lib = -lcublas -lcusolver -std=c++11

$(des):$(source)
	$(cc) -o $(des) $(source) $(lib)


clean:
	rm -rf $(des)
