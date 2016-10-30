require 'torch'
require 'PCANet'
local xlua = require 'xlua'
local util = require "util"
local nn = require 'nn'


function load_data(trsize,tesize)
	trsize = trsize or 50000
	tesize = tesize or 10000
	trsize = math.min(trsize,50000)
	tesize = math.min(tesize,10000)

	-- download dataset
	if not paths.dirp('cifar-10-batches-t7') then
	   print ("dataset not found, downloading the dataset...")
	   local www = 'http://torch7.s3-website-us-east-1.amazonaws.com/data/cifar-10-torch.tar.gz'
	   local tar = paths.basename(www)
	   print ("uncompressing...")
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
	ValData = {}
	ValData.data = trainData.data:index(1,val_idx:long())
	ValData.labels = trainData.labels:index(1,val_idx:long())
	print(ValData)

	trainData.data = trainData.data:index(1,tr_idx:long())
	trainData.labels = trainData.labels:index(1,tr_idx:long())

	trsize = tr_idx:size(1)
	valsize = val_idx:size(1)

	return trainData,ValData, testData
end


function train(options, trainData, ValData, testData)
	

	-- 1. Training (or loading) PCANet
	local timer = torch.Timer() -- the Timer starts to count now

	if not paths.filep("pcanet.t7") then
		pcanet = PCANet(options) -- create a PCANet instance
		timer:reset()
		print ('Training PCANet')
		pcanet:PCANet_train(trainData.data,options.MaxSamples)
		print('Time elapsed for training PCA Filters: ' .. timer:time().real .. ' seconds')
		print("saving the PCANet instance")
		torch.save("pcanet.t7",pcanet)
	else
		print ("loading PCA Filters")
		pcanet = torch.load("pcanet.t7")
	end
	-- print (pcanet.V)
	-- print (pcanet.HistBlockSize)

	-- try 1 img to get the dim of input of the NN
	local tmpf = pcanet:PCANet_FeaExt_single_input(torch.randn(1,3,32,32))
	local nInputDim = tmpf:size(1)


	-- 2. training NN
	-- if paths.filep("NN.t7") then 
	timer:reset()
	print ("training NN")
	local model = nn.Sequential()
	model:add(nn.Linear(nInputDim, 10)) 
	model:add(nn.LogSoftMax())
	local criterion = nn.ClassNLLCriterion() -- Negative log-likelihood criterion.
	-- 2. Select a batch of data and pass it through PCA filter first, then linear layer 
	print ("start momentum SGD")
	-- Go over the training data this number of times.
	local params, gradParams = model:getParameters() -- return 2 tensors
	local velocityParams = torch.zeros(gradParams:size())

	-- local train_features = pcanet:PCANet_FeaExt(trainData.data)
	-- local test_features = pcanet:PCANet_FeaExt(testData.data)

    for epoch = 1, options.max_epochs do
        local sum_loss = 0
        local correct = 0
        
        -- Run over the training set samples.
        model:training() -- turn on the training mode
        local n_batches = math.floor(trainData.data:size(1) / options.batch_size)
        for i = 1, n_batches do
            -- print ("1.1 Sample a batch")
            local img_batch_i = torch.Tensor(options.batch_size, 3, 32, 32)
            local labels = torch.Tensor(options.batch_size)

            for bi = 1, options.batch_size do
                local rand_id = torch.random(1, trainData.data:size(1))
                img_batch_i[bi] = trainData.data[i]
                labels[bi] = trainData.labels[rand_id]
            end
            inputs = pcanet:PCANet_FeaExt(img_batch_i)
            assert(inputs:size(2)==nInputDim,"dim not match")

            -- print ("1.2 Perform the forward pass (prediction mode).")
            local predictions = model:forward(inputs)
            
            -- print("1.3 Evaluate results")
            for i = 1, predictions:size(1) do
                local _, predicted_label = predictions[i]:max(1)
                if predicted_label[1] == labels[i] then correct = correct + 1 end
            end
            sum_loss = sum_loss + criterion:forward(predictions, labels)

            -- print("1.4 Perform the backward pass (compute derivatives)")
            -- This zeroes-out all the parameters inside the model pointed by variable params.
            model:zeroGradParameters()
            -- This internally computes the gradients with respect to the parameters pointed by gradParams.
            local gradPredictions = criterion:backward(predictions, labels)
            model:backward(inputs, gradPredictions)

            -- print ("1.5 Momentum update")
			-- v = mu * v - options.learning_rate * dx -- integrate velocity
			-- x += v -- integrate position
            velocityParams:mul(options.momentum_rate)  
            velocityParams:add(options.learning_rate, gradParams)
            params:add(-1, velocityParams)

            xlua.progress(i,n_batches)
            if i % 100 == 0 then  
                print(('train epoch=%d, iteration=%d, avg-loss=%.6f, avg-accuracy = %.2f')
                    :format(epoch, i, sum_loss / i, correct / (i * options.batch_size)))
            end
        end

        
        -- print("after each epoch, evaluate the accuracy for the val data")
        local validation_accuracy = 0
        model:evaluate() -- turn on the evaluation mode

        -- Perform the forward pass (prediction mode).
        local predictions = model:forward(pcanet:PCANet_FeaExt(ValData.data))
        -- evaluate results.
        for i = 1, predictions:size(1) do
            local _, predicted_label = predictions[i]:max(1)
            if predicted_label[1] == ValData.labels[i] then validation_accuracy = validation_accuracy + 1 end
        end

        validation_accuracy = validation_accuracy / (ValData.data:size(1))
        print(('\n validation accuracy at epoch = %d is %.4f'):format(epoch, validation_accuracy))


        print("learning rate decay")
        options.learning_rate = options.learning_rate * options.learning_rate_decay
        options.learning_rate = math.max(options.learning_rate,0.0001)

        if epoch % 5==0 then
        	torch.save("model.t7",model)
        end
        collectgarbage()

    end

    print('Time elapsed for training NN: ' .. timer:time().real .. ' seconds')


    print ("test time")
    local test_accuracy = 0
    model:evaluate() -- turn on the evaluation mode

    -- Perform the forward pass (prediction mode).
    local predictions = model:forward(pcanet:PCANet_FeaExt(testData.data))
    -- evaluate results.
    for i = 1, predictions:size(1) do
        local _, predicted_label = predictions[i]:max(1)
        if predicted_label[1] == testData.labels[i] then test_accuracy = test_accuracy + 1 end
    end

    test_accuracy = test_accuracy / (testData.data:size(1))
    print(test_accuracy)

end


function main(options)
	trainData, ValData, testData = load_data(500,100)
	-- print (trainData)
	train(options, trainData, ValData, testData)
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
cmd:option('-learning_rate',1,'starting learning rate')
cmd:option('-momentum_rate',0.9,'momentum rate')
cmd:option('-learning_rate_decay',0.5,'learning rate decay')
cmd:option('-decay_when',1,'decay if validation perplexity does not improve by more than this much')
cmd:option('-param_init', 0.05, 'initialize parameters at')
cmd:option('-batch_norm', 0, 'use batch normalization over input embeddings (1=yes)')
cmd:option('-batch_size',20,'number of sequences to train on in parallel')
cmd:option('-max_epochs',25,'number of full passes through the training data')


-- bookkeeping
cmd:option('-seed',3435,'torch manual random number generator seed')
cmd:option('-print_every',500,'how many steps/minibatches between printing out the loss')
cmd:option('-save_every', 5, 'save every n epochs')
cmd:option('-checkpoint_dir', 'cv', 'output directory where checkpoints get written')
cmd:option('-savefile','char','filename to autosave the checkpont to. Will be inside checkpoint_dir/')


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








