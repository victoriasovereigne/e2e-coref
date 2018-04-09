#!/bin/bash
export CUDA_VISIBLE_DEVICES="0,1"
export GPU="0"

echo $1

# evaluator does not use GPU --> set back to empty string

train_and_evaluate(){
	python singleton.py $1 >> notes/$1.txt
	python evaluator.py $1 >> notes/$1.txt
}

train_and_evaluate $1
train_and_evaluate $1

