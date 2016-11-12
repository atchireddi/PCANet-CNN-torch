require 'torch'
require 'PCANet'
local xlua = require 'xlua'
local util = require "util"
local nn = require 'nn'
require 'image'



function load_data(trsize,tesize)
    trsize = trsize or 50000
    tesize = tesize or 10000
    trsize = math.min(trsize,50000)
    tesize = math.min(tesize,10000)


    -- download dataset
    if not paths.dirp('cifar-10-batches-t7') then
   	print ("dataset not found, downloading and uncompressing the dataset...")
   	local www = 'http://torch7.s3-website-us-east-1.amazonaws.com/data/cifar-10-torch.tar.gz'
   	local tar = paths.basename(www)
   	os.execute('wget ' .. www .. '; '.. 'tar xvf ' .. tar)
    end


    print ("loading data...")
    -- load dataset
    trainData = {
   	data = torch.Tensor(50000, 3072),
   	labels = torch.Tensor(50000),
    }


    for i = 0,4 do
   	subset = torch.load('cifar-10-batches-t7/data_batch_' .. (i+1) .. '.t7', 'ascii')
   	trainData.data[{ {i*10000+1, (i+1)*10000} }] = subset.data:t()
   	trainData.labels[{ {i*10000+1, (i+1)*10000} }] = subset.labels
    end
    trainData.labels = trainData.labels + 1




    subset = torch.load('cifar-10-batches-t7/test_batch.t7', 'ascii')
    testData = {
   	data = subset.data:t():double(),
   	labels = subset.labels[1]:double(),
    }
    testData.labels = testData.labels + 1


    -- resize dataset (if using small version)
    trainData.data = trainData.data[{ {1,trsize} }]
    trainData.labels = trainData.labels[{ {1,trsize} }]


    testData.data = testData.data[{ {1,tesize} }]
    testData.labels = testData.labels[{ {1,tesize} }]


    -- reshape data
    trainData.data = trainData.data:reshape(trsize,3,32,32)
    testData.data = testData.data:reshape(tesize,3,32,32)


    -- create val data
    idx = torch.randperm(trsize)
    tr_idx = idx[{ {1,math.floor(trsize*0.9)}   }]
    val_idx = idx[{  {math.floor(trsize*0.9)+1,trsize}   }]
    valData = {}
    valData.data = trainData.data:index(1,val_idx:long())
    valData.labels = trainData.labels:index(1,val_idx:long())


    trainData.data = trainData.data:index(1,tr_idx:long())
    trainData.labels = trainData.labels:index(1,tr_idx:long())


    trsize = tr_idx:size(1)
    valsize = val_idx:size(1)
    return trainData, valData, testData


end


function train_PCA(options, trainData, valData, testData)
    -- 1. Training (or loading) PCANet
    local timer = torch.Timer() -- the Timer starts to count now


    if not paths.filep("model/pcanet.t7") then
     	 pcanet = PCANet(options) -- create a PCANet instance
     	 timer:reset()
     	 print ('Training PCANet')
     	 pcanet:PCANet_train(trainData.data,options.MaxSamples)
     	 print('Time elapsed for training PCA Filters: ' .. timer:time().real .. ' seconds')
     	 print("saving the PCANet instance")
     	 torch.save("model/pcanet.t7",pcanet)
    else
     	 print ("loading PCA Filters")
     	 pcanet = torch.load("model/pcanet.t7")
    end

    local V = pcanet.V[1]:clone()
    V:resize(options.NumFilters[1],3,options.kW,options.kH)
    torch.save("output/PCA_filters.t7",V)

    -- extract the features of train, val, and test data

    if not paths.filep("features/train_features.t7") then
        print ("extracting train features")
        timer:reset()
        local train_features = pcanet:PCANet_FeaExt(trainData.data)
        torch.save("features/train_features.t7",train_features)
        train_features = nil
        collectgarbage()
        print('Time elapsed for training features: ' .. timer:time().real .. ' seconds')
    end

    if not paths.filep("features/val_features.t7") then
        print ("extracting val features")
        timer:reset()
        local val_features = pcanet:PCANet_FeaExt(valData.data)
        torch.save("features/val_features.t7",val_features)
        val_features = nil
        collectgarbage()
        print('Time elapsed for val features: ' .. timer:time().real .. ' seconds')
    end

    if not paths.filep("features/test_features.t7") then
        print ("extracting test features")
        timer:reset()
        local test_features = pcanet:PCANet_FeaExt(testData.data)
        torch.save("features/test_features.t7",test_features)
        test_features = nil
        collectgarbage()
        print('Time elapsed for test features: ' .. timer:time().real .. ' seconds')
    end

    if not paths.filep("features/labels.t7") then
        -- output the labels
        labels = {trainData.labels, valData.labels, testData.labels}
        torch.save("features/labels.t7",labels)
    end
end


function train_classifier(options, train_features, val_features, test_features, labels)
  -- 
  local p = train_features:size(2) -- p is input dimension
  train_mean = torch.Tensor(p)
  train_std = torch.Tensor(p)
  for i = 1, p do
    train_mean[i] = train_features[{{}, {i}}]:mean()
    train_std[i] = train_features[{{}, {i}}]:std()
  end

  for i = 1, p do
    train_features[{{}, {i}}]:add(-train_mean[i]):div(train_std[i])
    val_features[{{}, {i}}]:add(-train_mean[i]):div(train_std[i])
    test_features[{{}, {i}}]:add(-train_mean[i]):div(train_std[i])
  end

    -- print(test_features:size())
    -- for i = 1, p do
    --  print (train_features[{{},{i}}]:mean(),train_features[{{},{i}}]:std())
    --  print (val_features[{{},{i}}]:mean(),train_features[{{},{i}}]:std())
    --  print (test_features[{{},{i}}]:mean(),train_features[{{},{i}}]:std())
    -- end
  local train_labels = labels[1]
  local val_labels = labels[2]
  local test_labels = labels[3]



    -- train
    local model = nn.Sequential()
    model:add(nn.Linear(p, 1024))
    model:add(nn.ReLU())
    model:add(nn.Dropout(0.5))
    model:add(nn.Linear(1024, 128))
    model:add(nn.ReLU())
    model:add(nn.Dropout(0.5))
    model:add(nn.Linear(128, 10))
    model:add(nn.LogSoftMax())
    -- model:cuda()
    local criterion = nn.ClassNLLCriterion() -- Negative log-likelihood criterion.
  

    local params, gradParams = model:getParameters() -- return 2 tensors

    local batch_size = options.batch_size or 64  -- The bigger the batch size the most accurate the gradients.
    local learning_rate = options.learning_rate or 0.001  -- This is the learning rate parameter often referred to as lambda.
    local momentum_rate = options.momentum_rate or 0.9
    local max_epochs = options.max_epochs or 500
    local learning_rate_decay = options.learning_rate_decay
    local velocityParams = torch.zeros(gradParams:size())

    preprocessFn = false

    local val_acc_list = {}

    -- Go over the training data this number of times.
    for epoch = 1, max_epochs do

        local sum_loss = 0
        local correct = 0
        
        -- Run over the training set samples.
        model:training() -- turn on the training mode
        local n_batches = math.floor(train_features:size(1) / batch_size)
        for i = 1, n_batches do
            
            -- 1. Sample a batch.
            local inputs = torch.Tensor(batch_size, p)
            local labels = torch.Tensor(batch_size)
            for bi = 1, batch_size do
                local rand_id = torch.random(1, train_features:size(1))
                if preprocessFn then
                    -- inputs[bi] = preprocessFn(train_features[rand_id])
                else
                    inputs[bi] = train_features[rand_id]
                end
                labels[bi] = train_labels[rand_id]
            end
            -- 2. Perform the forward pass (prediction mode).
            local predictions = model:forward(inputs)
            
            -- 3. Evaluate results.
            for i = 1, predictions:size(1) do
                local _, predicted_label = predictions[i]:max(1)
                if predicted_label[1] == labels[i] then correct = correct + 1 end
            end
            sum_loss = sum_loss + criterion:forward(predictions, labels)

            -- 4. Perform the backward pass (compute derivatives).
            -- This zeroes-out all the parameters inside the model pointed by variable params.
            model:zeroGradParameters()
            -- This internally computes the gradients with respect to the parameters pointed by gradParams.
            local gradPredictions = criterion:backward(predictions, labels)
            model:backward(inputs, gradPredictions)

            -- 5. Perform the SGD update.
            velocityParams:mul(momentum_rate)
            velocityParams:add(learning_rate, gradParams)
            params:add(-1, velocityParams)

            if i == n_batches then  -- Print
                print(('train epoch=%d, iteration=%d, avg-loss=%.4f, avg-accuracy = %.2f')
                    :format(epoch, i, sum_loss / i, correct / (i * batch_size)))
            end
            xlua.progress(i,n_batches)
        end

        -- save_every
        if epoch % options.save_every==0 then
          torch.save("model/Linear-".. epoch ..".t7", model)
        end


        -- Run over the validation set for evaluation.
        local validation_accuracy = 0
        local n_batches = val_features:size(1) / batch_size
        model:evaluate() -- turn on the evaluation mode
        for i = 1, n_batches do
            
            -- 1. Sample a batch.
            if preprocessFn then
                -- inputs = torch.Tensor(batch_size, 3, 224, 224)
            else
                inputs = torch.Tensor(batch_size, p)
            end
            local labels = torch.Tensor(batch_size)
            for bi = 1, batch_size do
                local rand_id = torch.random(1, val_features:size(1))
                if preprocessFn then
                    -- inputs[bi] = preprocessFn(val_features[rand_id])
                else
                    inputs[bi] = val_features[rand_id]
                end
                labels[bi] = val_labels[rand_id]
            end

            -- 2. Perform the forward pass (prediction mode).
            local predictions = model:forward(inputs)
            
            -- 3. evaluate results.
            for i = 1, predictions:size(1) do
                local _, predicted_label = predictions[i]:max(1)
                if predicted_label[1] == labels[i] then validation_accuracy = validation_accuracy + 1 end
            end
        end
        validation_accuracy = validation_accuracy / (n_batches * batch_size)
        print(('\nvalidation accuracy at epoch = %d is %.4f'):format(epoch, validation_accuracy))


      if epoch%10 ==0 and learning_rate>=0.00001 then 
        learning_rate = learning_rate * learning_rate_decay
        print ("learning_rate is ".. learning_rate .. "after" .. epoch .. "epochs") 
      end

      table.insert(val_acc_list,validation_accuracy)
    end -- for epoch = 1, max_epochs do

    print ("------------------------------------------test time------------------------------------------")
    local test_accuracy = 0
    model:evaluate() -- turn on the evaluation mode
    local inputs = test_features
        
    local predictions = model:forward(inputs)
    
    -- 3. evaluate results.
    for i = 1, predictions:size(1) do
        local _, predicted_label = predictions[i]:max(1)
        if predicted_label[1] == test_labels[i] then test_accuracy = test_accuracy + 1 end
    end

    test_accuracy = test_accuracy / test_features:size(1)
    print(('\ntest accuracy is %.4f'):format(test_accuracy))

    local results = {}
    results.test_accuracy = test_accuracy
    results.val_acc_list = val_acc_list
    torch.save("output/PCANet_result.t7",results)
end


function main(options)
    trainData, valData, testData = load_data()
    -- print (trainData)
    train_PCA(options, trainData, valData, testData)


    print ("loading features")
    local timer = torch.Timer() 
    timer:reset()
    train_features = torch.load("features/train_features.t7")
    val_features = torch.load("features/val_features.t7")
    test_features = torch.load("features/test_features.t7")
    labels = torch.load("features/labels.t7")
    print('Time elapsed for loading features: ' .. timer:time().real .. ' seconds')

    train_classifier(options, train_features, val_features, test_features, labels)
end




---------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a PCANet')
cmd:text()
cmd:text('Options')
-- data


-- PCANet params
cmd:option('-kW', 7, 'patch size in width dimensionality')
cmd:option('-kH', 7, 'patch size in height dimensionality')
cmd:option('-dW', 1, 'stride ')
cmd:option('-dH', 1, 'stride ')
cmd:option('-NumFilters','{8,8}', 'Number of PCA filters for each stage')
cmd:option('-HistBlockSize','{8,6}', '')
cmd:option('-BlkOverLapRatio',0.5, '')
cmd:option('MaxSamples',100000)


-- optimization
cmd:option('-learning_rate',0.001,'starting learning rate')
cmd:option('-momentum_rate',0.9,'momentum rate')
cmd:option('-learning_rate_decay',0.9,'learning rate decay')
-- cmd:option('-decay_when',1,'decay if validation perplexity does not improve by more than this much')
-- cmd:option('-param_init', 0.05, 'initialize parameters at')
cmd:option('-batch_norm', 0, 'use batch normalization over input embeddings (1=yes)')
cmd:option('-batch_size',64,'number of sequences to train on in parallel')
cmd:option('-max_epochs',100,'number of full passes through the training data')



-- bookkeeping
cmd:option('-seed',3435,'torch manual random number generator seed')
-- cmd:option('-print_every',500,'how many steps/minibatches between printing out the loss')
cmd:option('-save_every', 50, 'save every n epochs')
-- cmd:option('-checkpoint_dir', 'cv', 'output directory where checkpoints get written')
-- cmd:option('-savefile','char','filename to autosave the checkpont to. Will be inside checkpoint_dir/')




-- GPU/CPU
-- cmd:option('-gpuid', -1,'which gpu to use. -1 = use CPU')
-- cmd:option('-cudnn', 0,'use cudnn (1=yes). this should greatly speed up convolutions')
cmd:text()


-- parse input params
options = cmd:parse(arg)
torch.manualSeed(options.seed)


-- some housekeeping
loadstring('options.NumFilters = '.. options.NumFilters)()
loadstring('options.HistBlockSize = '.. options.HistBlockSize)()




main(options)






















