from sys import argv
from subprocess import call
import os
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed')
parser.add_argument('--main_dir')
parser.add_argument('--raw_data_dir')
parser.add_argument('--data_dir')
parser.add_argument('--data_set')
parser.add_argument('--percent_train')
parser.add_argument('--percent_dev')
parser.add_argument('--percent_test')
parser.add_argument('--pos_to_neg_ratio')
parser.add_argument('--model')
parser.add_argument('--optim_method')
parser.add_argument('--batch_size')
parser.add_argument('--dim')
parser.add_argument('--learning_rate')
parser.add_argument('--regularization')
parser.add_argument('--update_emb')
parser.add_argument('--emb_learning_rate')
parser.add_argument('--noise_level')
parser.add_argument('--epochs')

args = parser.parse_args()

print(args)
exit()

arg_counter=1
seed = argv[arg_counter]
arg_counter=arg_counter+1
main_dir = argv[arg_counter]
arg_counter=arg_counter+1
raw_data_dir=main_dir+'/'+argv[arg_counter]
arg_counter=arg_counter+1
data_dir=main_dir+'/'+argv[arg_counter]
arg_counter=arg_counter+1
data_set_raw_data_dir=raw_data_dir+'/'+argv[arg_counter]
data_set_data_dir=data_dir+'/'+argv[arg_counter]
arg_counter=arg_counter+1
percent_train=float(argv[arg_counter])
arg_counter=arg_counter+1
percent_dev=float(argv[arg_counter])
arg_counter=arg_counter+1
percent_test=float(argv[arg_counter])
arg_counter=arg_counter+1
percent_data=str(int(percent_train*100))

assert(percent_train+percent_dev+percent_test <= 1)

if not os.path.exists(data_set_data_dir):
    os.makedirs(data_set_data_dir)
    
pos_to_neg_ratio_search = re.search('\d_(\d)', argv[arg_counter], re.IGNORECASE)
if pos_to_neg_ratio_search:
    pos_to_neg_ratio = pos_to_neg_ratio_search.group(1)
 
pos_neg_ratio_dir=data_set_data_dir+'/'+argv[arg_counter]
percent_data_dir=pos_neg_ratio_dir+'/'+percent_data
train_dir=percent_data_dir+'/train'
dev_dir=percent_data_dir+'/dev'
test_dir=percent_data_dir+'/test'
avg_model_dir=percent_data_dir+'/avg'
lstm_model_dir=percent_data_dir+'/lstm'
bilstm_model_dir=percent_data_dir+'/bilstm'
  
if not os.path.exists(pos_neg_ratio_dir):
    os.makedirs(pos_neg_ratio_dir)
if not os.path.exists(percent_data_dir):
    os.makedirs(percent_data_dir)
if not os.path.exists(train_dir):
    os.makedirs(train_dir)
if not os.path.exists(dev_dir):
    os.makedirs(dev_dir)
if not os.path.exists(test_dir):
    os.makedirs(test_dir)
if not os.path.exists(avg_model_dir):
    os.makedirs(avg_model_dir)
if not os.path.exists(lstm_model_dir):
    os.makedirs(lstm_model_dir)
if not os.path.exists(bilstm_model_dir):
    os.makedirs(bilstm_model_dir)
    
call([main_dir+'/PrepData/bin/Release/./PrepData.exe',  data_set_raw_data_dir+'/walmart-amazon_perfectMapping.csv', data_set_raw_data_dir+'/walmart.csv' , data_set_raw_data_dir+'/amazon.csv', percent_data_dir+'/train.txt', percent_data_dir+'/dev.txt', percent_data_dir+'/test.txt', str(percent_train) , str(percent_dev), str(percent_test), pos_to_neg_ratio, seed])
os.chdir(main_dir)
call(['sh', 'fetch_and_preprocess.sh', percent_data_dir])

# th relatedness/main.lua --model bilstm --data_sub_folder walmart-amazon/1_1/10 --optim_method adagrad --batch_size 100 --dim 150 --sim_nhidden 50 --learning_rate 0.01 --regularization 1e-3 --update_emb true --emb_learning_rate 0.01 --noise_level 0.1 --epochs 40
