#!/bin/bash
problemsize=2048
workgroupsize=32

for((i = 1; i<32; i+=1))
do
	# Change work group size
	workgroupsize=`expr 32 \* $i` 
	echo "Work group size = " $workgroupsize
	./euler_v2 $workgroupsize $problemsize
	echo ""
done
