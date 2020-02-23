#!/bin/bash
#Shell script to monitor Tesla V100 performance and temperature while running experiments. When the temperature grows larger than the high threshold, this script stop executing experiments until the temperature drops to the (low threshold + 2).
#Author: Tao Zhang
#Date: 2018/12/22
#Usage: autotests.sh -m mode [-i gpuindex] [-f/-a]
#Valid mode: batched, streamed, based.
#-f: force GPU running without monitoring performance and temperature. This parameter is optional.
#-a: automatically runs GPU while monitoring performance and temperature. When the temperature grows larger than the high threshold, this script stop executing experiments until the temperature drops to the (low threshold + 2). This parameter is optional. It is enabled by default.
#output: two files, date-time-mode-result.txt containing the experiment results, date-time-mode-result.log containing the console output.
#example1: autotests.sh -m batched            run the batched tests with temperature monitoring and sleeping.
#example2: autotests.sh -m streamed -f        run the streamed tests and ignoring the GPU performance and temperature.
#set -x
export PATH="/usr/local/cuda-10.0/bin:/usr/local/MATLAB/R2017a/bin:/usr/local/magma:/opt/intel/lib/intel64:/opt/intel/mkl/lib/intel64:$PATH"
export export LD_LIBRARY_PATH="/usr/local/cuda-10.0/lib64:/opt/intel/compilers_and_libraries_2019.1.144/linux/mkl/lib/intel64_lin:/opt/intel/mkl/lib:/usr/local/magma/lib:$LD_LIBRARY_PATH"

echo "PATH: $PATH"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"




auto=1
cmode="batched"
iterations=1
highthreshold=75
lowthreshold=58
threshold=$highthreshold
temperatureoffset=1
sleeptime=10
gpuindex=0



while getopts ":m:i:fa" opt; do
	case $opt in 
		m) cmode=$OPTARG;;
		i) gpuindex=$OPTARG;;
    f) auto=0;;
    a) auto=1;;
    ?) echo "Parameter error"
        exit 1;;
  esac
done 

echo "cmode= $cmode"
if [ "$cmode"x != "batched"x -a "$cmode"x != "streamed"x -a "$cmode"x != "based"x ]; then
  echo "Usage: autotests.sh -m mode [-f/-a]"
  echo "Valid mode: batched, streamed, based."
  echo "-f: force GPU running without monitoring performance and temperature."
  echo "-a: automatically runs GPU while monitoring performance and temperature. This script enables this option by default."
  exit 1
fi

filestr="`pwd`/`date +%Y%m%d-%H%M`-$cmode-result.txt"
logstr="`pwd`/`date +%Y%m%d-%H%M`-$cmode-result.log"
touch $filestr
touch $logstr

echo "Starting $cmode mode test..."
echo "Starting $cmode mode test..." >> logstr

perf=`nvidia-smi -q -i $gpuindex -d PERFORMANCE | grep Performance | awk '{print $4}'`
temp=`nvidia-smi -q -i $gpuindex -d TEMPERATURE | grep Current | grep GPU | awk '{print $5}'`
echo "**********Tests begin: `date +%Y%m%d-%H%M%S`*********"
echo "**********Tests begin: `date +%Y%m%d-%H%M%S`*********" >> $filestr
echo "**********Tests begin: `date +%Y%m%d-%H%M%S`*********" >> $logstr
echo "**********Test results are saved into: $filestr"
echo "**********Test results are saved into: $filestr" >> $logstr
echo "GPU performance: $perf"
echo "GPU performance: $perf" >> $logstr
echo "Current temperature: $temp"
echo "Current temperature: $temp" >> $logstr
nvidia-smi >> $filestr
nvidia-smi >> $logstr

if [ $temp -lt $lowthreshold ]; then
	lowthreshold=$temp
fi

	echo "$cmode ++++++++++++++++++++"  >> $filestr
	echo "$cmode ++++++++++++++++++++"  >> $logstr
	   perf=`nvidia-smi -q -i $gpuindex -d PERFORMANCE | grep Performance | awk '{print $4}'`
     temp=`nvidia-smi -q -i $gpuindex -d TEMPERATURE | grep Current | grep GPU | awk '{print $5}'`
     
     if [ $auto -eq 1 ]; then
     	 echo "------> GPU [ $gpuindex ]: Auto = ON. Current performance: $perf, temperature: $temp."
       echo "------> GPU [ $gpuindex ]: Auto = ON. Current performance: $perf, temperature: $temp." >> $logstr
	     if [[ $perf == "P0" ]] && [[ $temp -lt $threshold ]]; then
	     	    threshold=$highthreshold
	     	    echo "Testing  $cmode"
	     	    echo "Testing  $cmode" >> $logstr
		       ./luhan $cmode  >> $filestr
		   else
		      if [ $lowthreshold -lt $temp ]; then
		      	temperatureoffset=`expr $temp - $lowthreshold`
		      	sleeptime=`expr $temperatureoffset \* 10`
		      	threshold=`expr $lowthreshold + 3`
		      fi
		   		echo "------> GPU [ $gpuindex ]: Temperature too high! Sleep $sleeptime seconds and try again..."
		   		echo "------> GPU [ $gpuindex ]: Temperature too high! Sleep $sleeptime seconds and try again..." >> $logstr
		   		sleep $sleeptime
		   fi
	   else
	     echo "------> GPU [ $gpuindex ]: Auto = OFF. Current performance: $perf, temperature: $temp."
       echo "------> GPU [ $gpuindex ]: Auto = OFF. Current performance: $perf, temperature: $temp." >> $logstr
	     echo "Testing  $cmode"
	     echo "Testing  $cmode" >> $logstr
	     ./luhan  $cmode  >> $filestr
	   fi
	   
echo "**********Tests end: `date +%Y%m%d-%H%M%S`*********"
echo "**********Tests end: `date +%Y%m%d-%H%M%S`*********" >> $logstr
echo "**********Tests end: `date +%Y%m%d-%H%M%S`*********" >> $filestr
exit 0
