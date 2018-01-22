--[[

  Training script for semantic relatedness prediction on the SICK dataset.

--]]

require('..')
require 'cunn'
require 'cudnn'
require 'rnn' 

-- Pearson correlation
function pearson(x, y)
  x = x - x:mean()
  y = y - y:mean()
  x = x:double()
  y = y:double()
  return x:dot(y) / (x:norm() * y:norm())
end

-- read command line arguments
local args = lapp [[
Training script for semantic relatedness prediction on the SICK dataset.
  -m,--model  (default dependency) Model architecture: [dependency, constituency, lstm, bilstm]
  -l,--layers (default 1)          Number of layers (ignored for Tree-LSTM)
  -d,--dim    (default 150)        LSTM memory dimension
  -e,--epochs (default 20)         Number of training epochs
  -s,--data_sub_folder (default sample)  data sub folder
  -p,--optim_method (default adam) optimization method
  -r,--learning_rate (default 0.05)  learning rate
  -b,--batch_size (default 25)  batch size
  -f,--folds (default 5) number of cross-validation folds
  -g,--regularization (default 1e-4) regularization
  -n,--sim_nhidden (default 50) size hidden layer
  -v,--update_emb (default false) update word vectors
  -w,--emb_learning_rate (default 1e-3) embeddings update rate
]]
torch.manualSeed(123)
local model_name, model_class
if args.model == 'dependency' then
  model_name = 'Dependency Tree LSTM'
  model_class = treelstm.TreeLSTMSim
elseif args.model == 'constituency' then
  model_name = 'Constituency Tree LSTM'
  model_class = treelstm.TreeLSTMSim
elseif args.model == 'lstm' then
  model_name = 'LSTM'
  model_class = treelstm.LSTMSim
elseif args.model == 'avg' then
  model_name = 'Average'
  model_class = treelstm.LSTMSim
elseif args.model == 'bilstm' then
  model_name = 'Bidirectional LSTM'
  model_class = treelstm.LSTMSim
end
local model_structure = args.model
header(model_name .. ' for Semantic Relatedness')


-- directory containing dataset files
local data_dir = 'data/dler/' .. args.data_sub_folder .. '/'
--local data_dir = 'data/dler/'
local vocab_dir = 'data/dler/'.. args.data_sub_folder .. '/'
--local vocab_dir = 'data/dler/'
local perf_dir = data_dir  .. args.model .. '/perf/'
local debug_dir = data_dir .. args.model .. '/debug/'


if lfs.attributes(debug_dir) == nil then
    lfs.mkdir(debug_dir)
end


if lfs.attributes(perf_dir) == nil then
  lfs.mkdir(perf_dir)
end

local file_idx = 1
while true do
  debug_file_path = string.format(debug_dir .. 'rel-%s.%dl.%dd.%dsimd.%.5flr.%.5freg.%swup.%5flrw.%sopt.%d.debug', args.model, args.layers, args.dim, args.sim_nhidden,args.learning_rate, args.regularization, args.update_emb, args.emb_learning_rate, args.optim_method,file_idx)
  if lfs.attributes(debug_file_path) == nil then
    break
  end
  file_idx = file_idx + 1
 end

debug_file = torch.DiskFile(debug_file_path, 'w')


local file_idx = 1
while true do
  train_save_data_path = string.format(data_dir .. 'rel-%s.%dl.%dd.%dsimd.%.5flr.%.5freg.%swup.%5flrw.%sopt.%d.%s.t7', args.model, args.layers, args.dim, args.sim_nhidden,args.learning_rate, args.regularization, args.update_emb, args.emb_learning_rate, args.optim_method,file_idx,'train_data')
  dev_save_data_path = string.format(data_dir .. 'rel-%s.%dl.%dd.%dsimd.%.5flr.%.5freg.%swup.%5flrw.%sopt.%d.%s.t7', args.model, args.layers, args.dim, args.sim_nhidden,args.learning_rate, args.regularization, args.update_emb, args.emb_learning_rate, args.optim_method,file_idx,'dev_data')
  test_save_data_path = string.format(data_dir .. 'rel-%s.%dl.%dd.%dsimd.%.5flr.%.5freg.%swup.%5flrw.%sopt.%d.%s.t7', args.model, args.layers, args.dim, args.sim_nhidden,args.learning_rate, args.regularization, args.update_emb, args.emb_learning_rate, args.optim_method,file_idx,'test_data')

  train_save_labels_path = string.format(data_dir .. 'rel-%s.%dl.%dd.%dsimd.%.5flr.%.5freg.%swup.%5flrw.%sopt.%d.%s.t7', args.model, args.layers, args.dim, args.sim_nhidden,args.learning_rate, args.regularization, args.update_emb, args.emb_learning_rate, args.optim_method,file_idx,'train_labels')
  dev_save_labels_path = string.format(data_dir .. 'rel-%s.%dl.%dd.%dsimd.%.5flr.%.5freg.%swup.%5flrw.%sopt.%d.%s.t7', args.model, args.layers, args.dim, args.sim_nhidden,args.learning_rate, args.regularization, args.update_emb, args.emb_learning_rate, args.optim_method,file_idx,'dev_labels')
  test_save_labels_path = string.format(data_dir .. 'rel-%s.%dl.%dd.%dsimd.%.5flr.%.5freg.%swup.%5flrw.%sopt.%d.%s.t7', args.model, args.layers, args.dim, args.sim_nhidden,args.learning_rate, args.regularization, args.update_emb, args.emb_learning_rate, args.optim_method,file_idx,'test_labels')
  
  if lfs.attributes(train_save_data_path) == nil then
    break
  end
  file_idx = file_idx + 1
 end



-- load vocab
local vocab = treelstm.Vocab(vocab_dir .. 'vocab-cased.txt')

-- load embeddings
print('loading word embeddings')
local emb_dir = 'data/glove/'
local emb_prefix = emb_dir .. 'glove.840B'
local emb_vocab, emb_vecs = treelstm.read_embedding(emb_prefix .. '.vocab', emb_prefix .. '.300d.th')
local emb_dim = emb_vecs:size(2)

-- use only vectors in vocabulary (not necessary, but gives faster training)
local num_unk = 0
local vecs = torch.CudaTensor(vocab.size, emb_dim)
for i = 1, vocab.size do
  local w = vocab:token(i)
  if emb_vocab:contains(w) then
    vecs[i] = emb_vecs[emb_vocab:index(w)]
  else
    num_unk = num_unk + 1
    vecs[i]:uniform(-0.05, 0.05)
  end
end
print('unk count = ' .. num_unk)
emb_vocab = nil
emb_vecs = nil
collectgarbage()

-- load datasets
print('loading datasets')
local train_dir = data_dir .. 'train/'
local dev_dir = data_dir .. 'dev/'
local test_dir = data_dir .. 'test/'

local is_blind_set = false
local train_dataset = treelstm.read_relatedness_dataset(train_dir, vocab, args.model, args.dataset, is_blind_set)
is_blind_set = false

printf('size of data = %d\n', train_dataset.size)

-- initialize model
local model = model_class{
  emb_learning_rate = args.emb_learning_rate,
  emb_vecs   = vecs,
  structure  = model_structure,
  num_layers = args.layers,
  mem_dim    = args.dim,
  learning_rate = args.learning_rate,
  batch_size = args.batch_size,
  reg = args.regularization, 
  sim_nhidden = args.sim_nhidden,
  update_emb = args.update_emb,
  debug_file = debug_file,
  optim_method = args.optim_method
}
-- number of epochs to train
local num_epochs = args.epochs
--number of folds to use
local num_folds = args.folds
-- print information
header('model configuration')
printf('max epochs = %d\n', num_epochs)
model:print_config()
-- train
local train_start = sys.clock()
local classes = {'1','2'}
local train_confusion = optim.ConfusionMatrix(classes)
local dev_confusion = optim.ConfusionMatrix(classes)
local test_confusion = optim.ConfusionMatrix(classes)
file_idx = 1
while true do
  perf_file_path = string.format(perf_dir .. '/rel-%s.%dl.%dd.%dsimd.%.5flr.%.5freg..%swup.%5flrw.%sopt.%d.pred', args.model, args.layers, args.dim, args.sim_nhidden,args.learning_rate, args.regularization, args.update_emb,args.emb_learning_rate, args.optim_method,file_idx)
  if lfs.attributes(perf_file_path) == nil then
    break
  end
  file_idx = file_idx + 1
end

local perf_file = torch.DiskFile(perf_file_path, 'w')
perf_file:writeString('epoch,train_F1,dev_F1,dev_precision,dev_recall\n')
local best_dev_model = model
local best_dev_score = 0
local best_dev_precision = 0
local best_dev_recall = 0
local best_dev_score_std = 0
local best_dev_precision_std = 0
local best_dev_recall_std = 0
local best_dev_epoch = 0
local train_splits = {}
local test_splits = {}
local leave_outs = {}
local fold_size = math.floor(train_dataset.size/num_folds)

for fold_no = 1, num_folds do
  local test_fold_start = (fold_no-1)*fold_size + 1
  local test_fold_end = fold_no*fold_size
  table.insert(leave_outs,{test_fold_start,test_fold_end})
end

header('Training model')
local data_to_save = {}
local folds_train_f1_scores = torch.DoubleTensor(num_folds)

local folds_dev_f1_scores = torch.DoubleTensor(num_folds)
local folds_dev_precisions = torch.DoubleTensor(num_folds)
local folds_dev_recalls = torch.DoubleTensor(num_folds)

-- local epochs_dev_f1_scores = torch.DoubleTensor(num_epochs)
-- local epochs_dev_precisions = torch.DoubleTensor(num_epochs)
-- local epochs_dev_recalls = torch.DoubleTensor(num_epochs)

local epoch_dev_f1_score = 0
local epoch_dev_precision = 0
local epoch_dev_recall = 0

local is_dev = false
for i = 1, num_epochs do
  local start = sys.clock()
  printf('-- epoch %d\n', i)
  for j = 1, num_folds do
    local saveData = false
    print('==========   Training Epoch: ' .. i .. ' , Fold: ' .. j .. '\n')
    model:train(train_dataset,leave_outs[j])
    printf('-- finished epoch in %.2fs\n', sys.clock() - start)
    is_blind_set = false
    print('==========   Eval Training Epoch: ' .. i .. ' , Fold: ' .. j .. '\n') 
    local train_predictions = model:predict_dataset(train_dataset, leave_outs[j], is_dev, train_confusion,is_blind_set)
    folds_train_f1_scores[j] = train_predictions[4]
    is_dev = true
    print('==========   Eval Dev Epoch: ' .. i .. ' , Fold: ' .. j .. '\n')
    local dev_predictions = model:predict_dataset(train_dataset, leave_outs[j], is_dev, dev_confusion,is_blind_set)
    is_dev = false

    folds_dev_precisions[j] = dev_predictions[2]
    folds_dev_recalls[j] = dev_predictions[3]
    folds_dev_f1_scores[j] = dev_predictions[4]
  end --folds
  local epoch_train_f1_score = folds_train_f1_scores:mean()
  -- epochs_dev_precisions[i] = folds_dev_precisions:mean()
  -- epochs_dev_recalls[i] = folds_dev_recalls:mean()
  -- epochs_dev_f1_scores[i] = folds_dev_f1_scores:mean()
  epoch_dev_precision = folds_dev_precisions:mean()
  epoch_dev_recall = folds_dev_recalls:mean()
  epoch_dev_f1_score = folds_dev_f1_scores:mean()

  -- if epochs_dev_f1_scores[i] > best_dev_score then
  if epoch_dev_f1_score > best_dev_score then
    best_dev_score = epoch_dev_f1_score
    best_dev_precision = epoch_dev_precision
    best_dev_recall = epoch_dev_recall
    best_dev_score_std = folds_dev_f1_scores:std()
    best_dev_precision_std = folds_dev_precisions:std()
    best_dev_recall_std = folds_dev_recalls:std()
    best_dev_model = model_class{
      emb_vecs = vecs,
      structure = model_structure,
      num_layers = args.layers,
      mem_dim    = args.dim,
      learning_rate = args.learning_rate,
      batch_size = args.batch_size,
      reg = args.regularization,
      sim_nhidden = args.sim_nhidden,
      update_emb = args.update_emb,
      emb_learning_rate = args.emb_learning_rate,
      debug_file = debug_file,
      optim_method = args.optim_method
    }
    best_dev_model.params:copy(model.params)
    best_dev_epoch = i
  end


  print('Epoch Training F1 Score: ' .. epoch_train_f1_score)
  -- print('Epoch Dev F1 Score:      ' .. epochs_dev_f1_scores[i])
  print('Epoch Dev F1 Score:      ' .. epoch_dev_f1_score)
  --perf_file:writeString(i .. ',' .. epoch_train_f1_score .. ','  .. epochs_dev_f1_scores[i].. ','  .. epochs_dev_precisions[i].. ','  .. epochs_dev_recalls[i] .. '\n')
  perf_file:writeString(i .. ',' .. epoch_train_f1_score .. ','  .. epoch_dev_f1_score.. ','  .. epoch_dev_precision.. ','  .. epoch_dev_recall .. '\n')
end --epochs

debug_file:close()
printf('finished in %.2fs\n', sys.clock() - train_start)
print('\nBest @Epoch: ' .. best_dev_epoch .. '\n')
-- print('\nPrecision: ' .. epochs_dev_precisions:max() .. '\n')
-- print('\nRecall: ' .. epochs_dev_recalls:max() .. '\n')
-- print('\nF1 Score: ' .. epochs_dev_f1_scores:max() .. '\n')
print('\nPrecision: ' .. best_dev_precision .. '\n')
print('\nRecall: ' .. best_dev_recall .. '\n')
print('\nF1 Score: ' .. best_dev_score .. '\n')

print('\nPrecision std: ' .. best_dev_precision_std .. '\n')
print('\nRecall std: ' .. best_dev_recall_std .. '\n')
print('\nF1 Score std: ' .. best_dev_score_std .. '\n')

perf_file:writeString('\nBest @Epoch: ' .. best_dev_epoch .. '\n')
-- perf_file:writeString('\nPrecision: ' .. epochs_dev_precisions:max() .. '\n')
-- perf_file:writeString('\nRecall: ' .. epochs_dev_recalls:max() .. '\n')
-- perf_file:writeString('\nF1 Score: ' .. epochs_dev_f1_scores:max() .. '\n')
perf_file:writeString('\nPrecision: ' .. best_dev_precision .. '\n')
perf_file:writeString('\nRecall: ' .. best_dev_recall .. '\n')
perf_file:writeString('\nF1 Score: ' .. best_dev_score .. '\n')

perf_file:writeString('\nPrecision std: ' .. best_dev_precision_std .. '\n')
perf_file:writeString('\nRecall std: ' .. best_dev_recall_std .. '\n')
perf_file:writeString('\nF1 Score std: ' .. best_dev_score_std .. '\n')



if lfs.attributes(treelstm.models_dir) == nil then
  lfs.mkdir(treelstm.models_dir)
end

-- get paths
file_idx = 1
local predictions_save_path, model_save_path
while true do
  model_save_path = string.format(treelstm.models_dir .. '/rel-%s.%dl.%dd.%dsimd.%.5flr.%.5freg.%swup.%5flrw.%sopt.%d.pred', args.model, args.layers, args.dim, args.sim_nhidden,args.learning_rate, args.regularization, args.update_emb, args.emb_learning_rate, args.optim_method,file_idx)
  if lfs.attributes(model_save_path) == nil then
    break
  end
  file_idx = file_idx + 1
end

-- write models to disk
print('writing model to ' .. model_save_path)
best_dev_model:save(model_save_path)

