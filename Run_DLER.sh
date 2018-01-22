#!/bin/bash
set -e

if [ $# -lt 6 ]; then
	echo "  "
	echo "USAGE $0 <Dataset> <first-table-name> <second-table-name> <do-sample> <do-preprocess> <do-full-sampling-control>"
	echo " "
	echo "INFO This script must be run from DLER dir"
	echo " "
	echo "INFO Other Fine Grained Parameters like data splitting, neg/pos ratio, columns to include, etc. can be controlled inside the script"
	echo " "
	echo "INFO e.g.: bash Run_DLER.sh Abt-Buy Abt Buy no no no"
	echo " "
	exit 1
fi

echo "  "
echo "Other Fine Grained Parameters like data splitting, neg/pos ratio, columns to include, etc. can be controlled inside the script"
echo "  "

DATA_DIR=$PWD/data
DATASET=$1
FIRST_TABLE=$2
SECOND_TABLE=$3
SAMPLE=$4											 #"no: train/dev/test data has been sampled"
FULL_CONTROL=$5										 #"no: preprocessing has been done, i.e. fetch_and_preprocess.sh has been run before and DATASET_DLER folder is ready"
PREPROCESS=$6                                        #"no:sample random negatives with same ratio on all splits and include all columns", "yes: sample all three types of negatives with different ratios across splits and choose which columns to include"
DATASET_DIR=$DATA_DIR/DataSets/$DATASET              #where the three main inputs (table1, table2, matches) are stored
SPLIT_DATASET_DIR=$DATA_DIR/SplitDataSets/$DATASET   #where data splits (train, dev, test) will be stored 
DATASET_DLER=$DATA_DIR/dler/$DATASET                 #same as SPLIT_DATA_SET_DIR, however this is were parsing and preprocessing will

#SIMPLE CONTROL SAMPLING
SEED=31
TRAIN_RATIO=0.25
DEV_RATIO=0.25
TEST_RAIO=0.5
NEG_TO_POS_RATIO=1

#FULL CONTROL SAMPLING
RATIO_NEG_TO_POS_TYPE_1=6    						#e.g. 2 neg examples for each pos example where the table one record of a random pos is part of the tuple
RATIO_NEG_TO_POS_TYPE_2=6							#e.g. 2 neg examples for each pos example where the table two record of a random pos is part of the tuple
RATIO_NEG_TO_POS_TYPE_3=6							#e.g. 2 neg examples for each pos example where both records are not part of any pos example

RATIO_OF_NEG_TO_INCLUDE_IN_TRAINING=0.025 			#off all samples negative examples for training, what ratio do you want to keep (class balancing)

COLUMNS_TO_INCLUDE_FROM_FIRST="1,2,3" 							#only include these columns as part of the record, column 0 is assumed to be the id
COLUMNS_TO_INCLUDE_FROM_SECOND="1,2,4" 							#only include these columns as part of the record, column 0 is assumed to be the id



#TRAINING PARAMETERS
NET_ARCH='avg'                                  # avg | lstm | bilstm                      
OPTIM_METHOD='adam'                                # sgd | adam | adagrad
BATCH_SIZE=16
RNN_DIM=150                                        #rnn units
HIDDEN_LAYER_SIZE=50                              
LEARNING_RATE=0.01
REGULARIZATION=1e-3
END_TO_END_LEARNING=false                          #updates embeddings for current training session
EMBEDDING_UPDATE_RATE=0.01
NOISE_LEVEL=0
NUM_EPOCHS=20	

if [ "$SAMPLE" = "yes" ]
then
	echo 'building required dir structure ..'
	rm -rf $SPLIT_DATASET_DIR
	mkdir $SPLIT_DATASET_DIR
	rm -rf $DATASET_DLER
	mkdir $DATASET_DLER
	mkdir $DATASET_DLER/avg
	mkdir $DATASET_DLER/lstm
	mkdir $DATASET_DLER/bilstm
	mkdir $DATASET_DLER/avg/debug
	mkdir $DATASET_DLER/lstm/debug
	mkdir $DATASET_DLER/bilstm/debug


	if [ "$FULL_CONTROL" = "yes" ]
	then
		echo 'full control sampling & preparing the data for parsing ..'
		$PWD/PrepData/./PrepMagellan.exe $DATASET_DIR/$DATASET"_perfectMapping.csv" "$DATASET_DIR/$FIRST_TABLE.csv" "$DATASET_DIR/$SECOND_TABLE.csv" \
										 "$SPLIT_DATASET_DIR/train.csv" "$SPLIT_DATASET_DIR/dev.csv" "/$SPLIT_DATASET_DIR/test.csv" \
										 "$DATASET_DLER/train.txt" "$DATASET_DLER/dev.txt" "$DATASET_DLER/test.txt" \
										 $TRAIN_RATIO $DEV_RATIO $TEST_RAIO \
										 $RATIO_NEG_TO_POS_TYPE_1 $RATIO_NEG_TO_POS_TYPE_2 $RATIO_NEG_TO_POS_TYPE_3 \
										 $RATIO_OF_NEG_TO_INCLUDE_IN_TRAINING \
										 $SEED $COLUMNS_TO_INCLUDE_FROM_FIRST $COLUMNS_TO_INCLUDE_FROM_SECOND
	    echo " "
	else
		echo 'simle sampling & preparing the data for parsing ..'								 
		$PWD/PrepData/./PrepData.exe $DATASET_DIR/$DATASET"_perfectMapping.csv" "$DATASET_DIR/$FIRST_TABLE.csv" "$DATASET_DIR/$SECOND_TABLE.csv" \
										 "$DATASET_DLER/train.txt" "$DATASET_DLER/dev.txt" "$DATASET_DLER/test.txt" \
										 $TRAIN_RATIO $DEV_RATIO $TEST_RAIO $NEG_TO_POS_RATIO $SEED
		echo " "
	fi
else
	echo "No Sampling was done!"
fi

if [ "$PREPROCESS" = "yes" ]
then
	echo 'preprocessing and parsing the data for model building and testing ..'
	sh fetch_and_preprocess.sh $DATASET_DLER
	echo " "
fi

echo 'building a model and test it ..'
th relatedness/main.lua --model $NET_ARCH --data_sub_folder $DATASET_DLER \
						--optim_method $OPTIM_METHOD --batch_size $BATCH_SIZE --dim $RNN_DIM --sim_nhidden $HIDDEN_LAYER_SIZE \
						--learning_rate $LEARNING_RATE --regularization $REGULARIZATION --update_emb $EMBEDDING_UPDATE_RATE \
						--emb_learning_rate $EMBEDDING_UPDATE_RATE --noise_level $NOISE_LEVEL --epochs $NUM_EPOCHS