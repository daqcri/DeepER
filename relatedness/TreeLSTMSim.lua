--[[

  Semantic relatedness prediction using Tree-LSTMs.

--]]
function isnan(x) return x ~= x end
 

local TreeLSTMSim = torch.class('treelstm.TreeLSTMSim')

function TreeLSTMSim:__init(config)
  self.mem_dim       = config.mem_dim       or 150
  self.learning_rate = config.learning_rate or 0.05
  self.emb_learning_rate = config.emb_learning_rate or 0.05
  self.batch_size    = config.batch_size    or 25
  self.reg           = config.reg           or 1e-4
  self.structure     = config.structure     or 'dependency' -- {dependency, constituency}
  self.sim_nhidden   = config.sim_nhidden   or 50
  self.update_emb = config.update_emb or 'false'
  self.debug_file = config.debug_file
  -- word embedding
  if self.update_emb == 'true' then
    self.update_emb = true
  else
    self.update_emb = false
  end
  
  self.emb_dim = config.emb_vecs:size(2)
  if self.update_emb == true then
    self.emb = nn.LookupTable(config.emb_vecs:size(1), self.emb_dim)
    self.emb:cuda()
    self.emb.weight:copy(config.emb_vecs:cuda())
  else
    self.emb_vecs = config.emb_vecs
  end
  -- number of similarity rating classes
  self.num_classes = 2

  -- optimizer configuration
  self.optim_state = { learningRate = self.learning_rate }

  self.criterion = nn.ModuleCriterion(nn.ClassNLLCriterion(), nn.Log())
  self.criterion:cuda()
  -- initialize tree-lstm model
  local treelstm_config = {
    in_dim = self.emb_dim,
    mem_dim = self.mem_dim,
    gate_output = false,
  }
  


  if self.structure == 'dependency' then
    self.treelstm = treelstm.ChildSumTreeLSTM(treelstm_config)
  elseif self.structure == 'constituency' then
    self.treelstm = treelstm.BinaryTreeLSTM(treelstm_config)
  else
    error('invalid parse tree type: ' .. self.structure)
  end

  -- similarity model
  self.sim_module = self:new_sim_module()
  local modules = nn.Parallel()
    :add(self.treelstm)
    :add(self.sim_module)
  self.params, self.grad_params = modules:getParameters()
end

function TreeLSTMSim:new_sim_module()
  local vecs_to_input
  local lvec = nn.Identity()()
  local rvec = nn.Identity()()
  local mult_dist = nn.CMulTable(){lvec, rvec}
  local add_dist = nn.Abs()(nn.CSubTable(){lvec, rvec})
  local vec_dist_feats = nn.JoinTable(1){mult_dist, add_dist}
  vecs_to_input = nn.gModule({lvec, rvec}, {vec_dist_feats})

   -- define similarity model architecture
  local sim_module = nn.Sequential()
    :add(vecs_to_input)
    :add(nn.Linear(2 * self.mem_dim, self.sim_nhidden))
    :add(nn.Sigmoid())    -- does better than tanh
    :add(nn.Linear(self.sim_nhidden, self.num_classes))
    :add(nn.SoftMax())
  sim_module:cuda()
  return sim_module
end

function TreeLSTMSim:train(dataset)
  self.treelstm:training()
  local indices = torch.randperm(dataset.size)
  local zeros = torch.CudaTensor(self.mem_dim):zero()
  for i = 1, dataset.size, self.batch_size do
    xlua.progress(i, dataset.size)
    local batch_size = math.min(i + self.batch_size - 1, dataset.size) - i + 1

    -- get target distributions for batch
    local targets = torch.CudaTensor(batch_size, 1):zero()
    for j = 1, batch_size do
      local idx = indices[i + j - 1]
      targets[j] = dataset.labels[idx]
    end

    local feval = function(x)
      self.grad_params:zero()
      if self.update_emb == true then
        self.emb:zeroGradParameters()
      end
      local loss = 0
      for j = 1, batch_size do
        local idx = indices[i + j - 1]
        
        local ltree, rtree = dataset.ltrees[idx], dataset.rtrees[idx]
        local lsent, rsent = dataset.lsents[idx], dataset.rsents[idx]

        if self.update_emb == true then
          self.emb:forward(lsent)
          linputs = torch.CudaTensor(self.emb.output:size()):copy(self.emb.output)
          rinputs = self.emb:forward(rsent)
        else
          linputs = self.emb_vecs:index(1, lsent:long()):cuda()
          rinputs = self.emb_vecs:index(1, rsent:long()):cuda()
        end        

        -- self.emb:forward(lsent)
        -- local linputs = torch.CudaTensor(self.emb.output:size()):copy(self.emb.output)
        -- local rinputs = self.emb:forward(rsent)
        
        -- get sentence representations
        local lrep = self.treelstm:forward(ltree, linputs)[2]
        local rrep = self.treelstm:forward(rtree, rinputs)[2]
        

        -- compute relatedness
        local output = self.sim_module:forward{lrep, rrep}

        -- compute loss and backpropagate
        local example_loss = self.criterion:forward(output, targets[j])
        loss = loss + example_loss
        local sim_grad = self.criterion:backward(output, targets[j])
        local rep_grad = self.sim_module:backward({lrep, rrep}, sim_grad)
        
        local linput_grads = self.treelstm:backward(ltree, linputs, {zeros, rep_grad[1]})       
        local rinput_grads = self.treelstm:backward(rtree, rinputs, {zeros, rep_grad[2]})
        
        
        if self.update_emb == true then
          self.emb:backward(rsent, rinput_grads)
          self.emb:forward(lsent)
          self.emb:backward(lsent, linput_grads)
        end
        -- local emb_grad_sum = self.emb.gradWeight:sum()
        -- if isnan(emb_grad_sum) == true or emb_grad_sum == math.huge or emb_grad_sum == -math.huge then
        --   local nan_mask = self.emb.gradWeight:ne(self.emb.gradWeight)
        --   local inf_mask = self.emb.gradWeight:eq(math.huge)
        --   local neg_inf_mask = self.emb.gradWeight:eq(-math.huge)  
        --   self.emb.gradWeight[nan_mask] = 0          
        -- end
      end
      
      loss = loss / batch_size    
      self.grad_params:div(batch_size)  
      if self.update_emb == true then   
        self.emb.gradWeight:div(batch_size)
        self.emb:updateParameters(self.emb_learning_rate)
      end
      
      --self.debug_file:writeString(i .. ', ' .. self.emb.gradWeight:norm() .. ', ' .. self.emb.weight:norm() .. '\n')

      -- regularization
      loss = loss + 0.5 * self.reg * self.params:norm() ^ 2
      self.grad_params:add(self.reg, self.params)
      return loss, self.grad_params
    end

    optim.adagrad(feval, self.params, self.optim_state)
  end
  xlua.progress(dataset.size, dataset.size)
end

-- Predict the similarity of a sentence pair.
function TreeLSTMSim:predict(ltree, rtree, lsent, rsent)
  local linputs = nil
  local rinputs = nil
  local lrep = nil
  local rrep = nil
  if self.update_emb == true then
    self.emb:forward(lsent)
    linputs = torch.CudaTensor(self.emb.output:size()):copy(self.emb.output)
    rinputs = self.emb:forward(rsent)
  else
    linputs = self.emb_vecs:index(1, lsent:long()):cuda()
    rinputs = self.emb_vecs:index(1, rsent:long()):cuda()
  end
  local lrep = self.treelstm:forward(ltree, linputs)[2]
  local rrep = self.treelstm:forward(rtree, rinputs)[2]
  local output = self.sim_module:forward{lrep, rrep}
  self.treelstm:clean(ltree)
  self.treelstm:clean(rtree)
  return output
end

-- Produce similarity predictions for each sentence pair in the dataset.
function TreeLSTMSim:predict_dataset(dataset, confusion, is_blind_set)
  self.treelstm:evaluate()
  local pred_loss = 0
  local predictions = torch.CudaTensor(dataset.size,2)
  for i = 1, dataset.size do
    xlua.progress(i, dataset.size)
    local ltree, rtree = dataset.ltrees[i], dataset.rtrees[i]
    local lsent, rsent = dataset.lsents[i], dataset.rsents[i]
    predictions[i] = self:predict(ltree, rtree, lsent, rsent)
    if is_blind_set == false then
      local sim = dataset.labels[i]
      local pred_instance_loss = self.criterion:forward(predictions[i], sim)
      pred_loss = pred_loss + pred_instance_loss
      if confusion~=nil then
        confusion:add(predictions[i], sim)
      end
    end
  end
  return {predictions, (pred_loss/dataset.size)}
end

function TreeLSTMSim:print_config()
  local num_params = self.params:size(1)
  local num_sim_params = self:new_sim_module():getParameters():size(1)
  printf('%-25s = %d\n',   'num params', num_params)
  printf('%-25s = %d\n',   'num compositional params', num_params - num_sim_params)
  printf('%-25s = %d\n',   'word vector dim', self.emb_dim)
  printf('%-25s = %s\n',   'udpate word vector', self.update_emb)
  printf('%-25s = %.2e\n', 'word vector learning rate', self.emb_learning_rate)
  printf('%-25s = %d\n',   'Tree-LSTM memory dim', self.mem_dim)
  printf('%-25s = %.2e\n', 'regularization strength', self.reg)
  printf('%-25s = %d\n',   'minibatch size', self.batch_size)
  printf('%-25s = %.2e\n', 'learning rate', self.learning_rate)
  printf('%-25s = %s\n',   'parse tree type', self.structure)
  printf('%-25s = %d\n',   'sim module hidden dim', self.sim_nhidden)
end

--
-- Serialization
--

function TreeLSTMSim:save(path)
  local embeddings = nil
  if self.update_emb == true then
    embeddings = self.emb.weight:float()
  else
    embeddings = self.emb_vecs:float()
  end
  
  local config = {
    batch_size    = self.batch_size,
    emb_vecs      = embeddings,
    learning_rate = self.learning_rate,
    emb_learning_rate = self.emb_learning_rate,
    mem_dim       = self.mem_dim,
    sim_nhidden   = self.sim_nhidden,
    reg           = self.reg,
    structure     = self.structure,
  }

  torch.save(path, {
    params = self.params,
    config = config,
  })
end

function TreeLSTMSim.load(path)
  local state = torch.load(path)
  local model = treelstm.TreeLSTMSim.new(state.config)
  model.params:copy(state.params)
  return model
end
