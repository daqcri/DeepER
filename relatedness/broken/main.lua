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
local dev_dataset = treelstm.read_relatedness_dataset(dev_dir, vocab, args.model, args.dataset, is_blind_set)
is_blind_set = false
local test_dataset = treelstm.read_relatedness_dataset(test_dir, vocab, args.model, args.dataset, is_blind_set)






printf('num train = %d\n', train_dataset.size)
printf('num dev   = %d\n', dev_dataset.size)
printf('num test  = %d\n', test_dataset.size)

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

-- print information
header('model configuration')
printf('max epochs = %d\n', num_epochs)
model:print_config()

-- train
local train_start = sys.clock()

local best_dev_score = 0
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
perf_file:writeString('epoch,train_nll,dev_nll\n')


local best_dev_model = model
local best_dev_epoch = 0
local train_to_save_data = torch.FloatTensor(train_dataset.size,600)
local dev_to_save_data = torch.FloatTensor(dev_dataset.size,600)
local test_to_save_data = torch.FloatTensor(test_dataset.size,600)
local train_to_save_labels = torch.FloatTensor(train_dataset.size)
local dev_to_save_labels = torch.FloatTensor(dev_dataset.size)
local test_to_save_labels = torch.FloatTensor(test_dataset.size)
local do_save = false

header('Training model')


local data_to_save = {}
for i = 1, num_epochs do
  local start = sys.clock()
  printf('-- epoch %d\n', i)
  -- if i == 1 then
  --   do_save = true
  -- end


  local indices = torch.randperm(dataset.size)
  model:train(train_dataset,fno,indices)
  printf('-- finished epoch in %.2fs\n', sys.clock() - start)
  is_blind_set = false
   
  local train_predictions = model:predict_dataset(train_dataset, train_confusion,is_blind_set, do_save, train_to_save_data, train_to_save_labels)
  local train_score = train_predictions[4]
  print('train confusion: \n')
  print(train_confusion)
  print("train F1 Score: " .. train_score)
  train_confusion:zero()
  local dev_predictions = model:predict_dataset(dev_dataset, dev_confusion,is_blind_set, do_save, dev_to_save_data, dev_to_save_labels)
  local dev_score = dev_predictions[4]
  print('dev confusion: \n')
  print(dev_confusion)
  print("dev F1 Score: " .. dev_score)
  perf_file:writeString(i .. ',' .. train_score ..',' .. dev_score .. '\n')
  dev_confusion:zero()

  if dev_score > best_dev_score then
    best_dev_score = dev_score
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
  do_save = false
end
torch.save(train_save_data_path, train_to_save_data)
torch.save(dev_save_data_path, dev_to_save_data)

torch.save(train_save_labels_path, train_to_save_labels)
torch.save(dev_save_labels_path, dev_to_save_labels)

do_save = false
debug_file:close()
printf('finished training in %.2fs\n', sys.clock() - train_start)

-- evaluate
header('Evaluating on test set')
printf('-- using model with dev F1 Score = %.4f\n', best_dev_score)
is_blind_set = false  
local test_predictions = best_dev_model:predict_dataset(test_dataset, test_confusion,is_blind_set,do_save, test_to_save_data, test_to_save_labels)

torch.save(test_save_data_path, test_to_save_data)
torch.save(test_save_labels_path, test_to_save_labels)

local test_precision = test_predictions[2]
local test_recall = test_predictions[3]
local test_f1_score = test_predictions[4]
print('test confusion: \n')
print(test_confusion)
print("test F1 Score: " .. test_f1_score)
perf_file:writeString('\nBest Dev Epoch: ' .. best_dev_epoch .. '\n')
perf_file:writeString('\nTest Precision: ' .. test_precision .. '\n')
perf_file:writeString('\nTest Recall: ' .. test_recall .. '\n')
perf_file:writeString('\nTest F1 Score: ' .. test_f1_score .. '\n')


test_confusion:zero()

-- create predictions and model directories if necessary
if lfs.attributes(treelstm.predictions_dir) == nil then
  lfs.mkdir(treelstm.predictions_dir)
end

if lfs.attributes(treelstm.models_dir) == nil then
  lfs.mkdir(treelstm.models_dir)
end

-- get paths
file_idx = 1
local predictions_save_path, model_save_path
while true do
  predictions_save_path = string.format(treelstm.predictions_dir .. '/rel-%s.%dl.%dd.%dsimd.%.5flr.%.5freg.%swup.%5flrw.%sopt.%d.pred', args.model, args.layers, args.dim, args.sim_nhidden,args.learning_rate, args.regularization, args.update_emb, args.emb_learning_rate, args.optim_method,file_idx)
  model_save_path =             string.format(treelstm.models_dir .. '/rel-%s.%dl.%dd.%dsimd.%.5flr.%.5freg.%swup.%5flrw.%sopt.%d.pred', args.model, args.layers, args.dim, args.sim_nhidden,args.learning_rate, args.regularization, args.update_emb, args.emb_learning_rate, args.optim_method,file_idx)
  if lfs.attributes(predictions_save_path) == nil and lfs.attributes(model_save_path) == nil then
    break
  end
  file_idx = file_idx + 1
end

-- write predictions to disk
local predictions_file = torch.DiskFile(predictions_save_path, 'w')
print('writing predictions to ' .. predictions_save_path)
predictions_file:writeString('best model dev score = ' .. best_dev_score .. '\n\n')

 


predictions_file:writeString('word vector dim = ' .. emb_dim .. '\n')
predictions_file:writeString('Tree-LSTM memory dim = ' .. args.dim .. '\n')
predictions_file:writeString('regularization strength = ' .. args.regularization .. '\n')
predictions_file:writeString('minibatch size = ' .. args.batch_size .. '\n')
predictions_file:writeString('learning rate = ' .. args.learning_rate .. '\n')
predictions_file:writeString('word vector learning rate = ' .. args.emb_learning_rate .. '\n')
predictions_file:writeString('model type = ' .. args.model .. '\n')
predictions_file:writeString('sim module hidden dim = ' .. args.sim_nhidden .. '\n\n')

predictions_file:writeString('test_id,is_duplicate\n')
for i = 1, test_predictions[1]:size(1) do
  predictions_file:writeString(test_dataset.ids[i] .. ',' ..   test_predictions[1][i][2] .. '\n')
end
predictions_file:close()

-- write models to disk
print('writing model to ' .. model_save_path)
best_dev_model:save(model_save_path)

-- to load a saved model
-- local loaded = model_class.load(model_save_path)
