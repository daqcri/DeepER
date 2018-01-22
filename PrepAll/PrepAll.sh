MAIN_DIR="/home/me/Documents/Code/torch/treelstm/origin/treelstm-origin-gpu-dler"

python3 PrepAll.py  --seed 123 \
				    --main_dir $MAIN_DIR \
				    --raw_data rawData \
					--data_dir data/DeepER \
					--data_set walmart-amazon \
					--percent_train 0.1 \
					--percent_dev 0.1 \
					--percent_test 0.1 \
					--pos_to_neg_ratio 1_1 \
					--model bilstm \
					--optim_method adagrad \
					--batch_size 100 \
					--dim 150 \
					--learning_rate 0.01 \
					--regularization 0.001 \
					--update_emb true \
					--emb_learning_rate 0.01 \
					--noise_level 0.1 \
					--epochs 40
					