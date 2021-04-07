#!/bin/bash

echo "Empezamos el experimento..."
echo ""

rm -r exp1Arc/

mkdir exp1Arc/

for file in ../../../experimento1/Test-data/*
do
	rm -r dataset/test/test/*

	cp -r $file/out/* dataset/test/test/

	echo ""
	echo $file
	echo ""  

	echo "" >> "exp1Arc/exp1Arc.txt"
	echo $file >> "exp1Arc/exp1Arc.txt"
	echo "" >> "exp1Arc/exp1Arc.txt"

	python3 classifier_mio.py CLASSIFY dataset/test/ classifier/mi_classifier.plk >> "exp1Arc/exp1Arc.txt"

done
