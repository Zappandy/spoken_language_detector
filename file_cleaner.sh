#!/bin/bash
# sources: https://stackoverflow.com/questions/6482377/check-existence-of-input-argument-in-a-bash-shell-script
# https://www.educba.com/if-condition-in-shell-script/
# 1 must be dir where flac files are: /media/andres/2D2DA2454B8413B5/test/test  
# 2 must be dir where es flac files will be: /media/andres/2D2DA2454B8413B5/test/data_sample
#if 5>2;
if [ $# -eq 2 ];  # important to leave spaces. checking for 2 args
then
	#og_dir=`find $1 -name '*.flac' -print`
	#og_dir=`find $1 -iregex '.*.flac' -print`
	og_dir=`find $1 -regextype sed -regex '.*/es.*' -print`  # find matches whole path ergo why we need .*/
else
	echo "Stopping shell script..."
	exit
fi

#if [ -z "$og_dir" ];  # does not work and 2 we already exit before so no need, unless empty? idk
for file in $og_dir
do 
	echo $file
	cp -a $file $2
done

zip -r test_set_es.zip $2

#echo $og_dir
