require 'nn'
require 'xlua'

-- The default tensor type in Torch is DoubleTensor, but we generally only need Float precision.
torch.setdefaulttensortype('torch.FloatTensor')

local model = nn.Sequential()
model:add(nn.SpatialConvolution(3, 32, 5, 5))  -- 3 input channels, 8 output channels (8 filters), 5x5 kernels.
model:add(nn.SpatialBatchNormalization(32, 1e-3))  -- BATCH NORMALIZATION LAYER.
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(2, 2, 2, 2)) -- Max pooling in 2 x 2 area.
model:add(nn.SpatialConvolution(32, 64, 5, 5))  -- 8 input channels, 16 output channels (16 filters), 5x5 kernels.
model:add(nn.SpatialBatchNormalization(64, 1e-3))  -- BATCH NORMALIZATION LAYER.
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(2, 2, 2, 2))  -- Max pooling in 2 x 2 area.
model:add(nn.SpatialConvolution(64, 16, 1, 1))  -- 8 input channels, 16 output channels (16 filters), 5x5 kernels.
model:add(nn.SpatialBatchNormalization(16, 1e-3))  -- BATCH NORMALIZATION LAYER.
model:add(nn.ReLU())
model:add(nn.View(16*5*5))    -- Vectorize the output of the convolutional layers.
model:add(nn.Linear(16*5*5, 1024))
model:add(nn.ReLU())
model:add(nn.Dropout(0.5))
model:add(nn.Linear(1024, 128))
model:add(nn.ReLU())
model:add(nn.Dropout(0.5))
model:add(nn.Linear(128, 10))
model:add(nn.LogSoftMax())

local criterion = nn.ClassNLLCriterion() -- Negative log-likelihood criterion.


function load_data(trsize,tesize)
	trsize = trsize or 50000  -- the input size is 3 x 32 x 32
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
	valData = {}
	valData.data = trainData.data:index(1,val_idx:long())
	valData.labels = trainData.labels:index(1,val_idx:long())
	print(ValData)

	trainData.data = trainData.data:index(1,tr_idx:long())
	trainData.labels = trainData.labels:index(1,tr_idx:long())

	trsize = tr_idx:size(1)
	valsize = val_idx:size(1)


	-- preprocess 
	trainData.normdata = trainData.data:clone():float()
	valData.normdata = valData.data:clone():float()
	testData.normdata = testData.data:clone():float()

	cifarMean = {trainData.normdata[{{}, {1}, {}, {}}]:mean(),
             trainData.normdata[{{}, {2}, {}, {}}]:mean(),
             trainData.normdata[{{}, {3}, {}, {}}]:mean()}

	cifarStd = {trainData.normdata[{{}, {1}, {}, {}}]:std(),
            trainData.normdata[{{}, {2}, {}, {}}]:std(),
            trainData.normdata[{{}, {3}, {}, {}}]:std()}

    -- Print the mean and std value for each channel.
	print(cifarMean)
	print(cifarStd)

    if not paths.dirp('cifar-10-batches-t7') then 
        cifarInfo = {}
        cifarInfo.mean = cifarMean
        cifarInfo.std = cifarStd
        torch.save("model/cifarInfo.t7",cifarInfo)
    end
	-- Now normalize the training and validation data.
	for i  = 1, 3 do
	    -- Subtracting the mean on each channel makes the values roughly between -128 and 128.
	    trainData.normdata[{{}, {i}, {}, {}}]:add(-cifarMean[i])
	    valData.normdata[{{}, {i}, {}, {}}]:add(-cifarMean[i])
	    testData.normdata[{{}, {i}, {}, {}}]:add(-cifarMean[i])
	    -- Dividing the std on each channel makes the values roughly between -1 and 1.
	    trainData.normdata[{{}, {i}, {}, {}}]:div(cifarStd[i])
	    valData.normdata[{{}, {i}, {}, {}}]:div(cifarStd[i])
	    testData.normdata[{{}, {i}, {}, {}}]:div(cifarStd[i])
	end

	trainData.data = nil
	testData.data = nil
	valData.data = nil

	return trainData, valData, testData
	-- return trainData, testData
end

function trainModel(model, options, trainData, valData, testData, preprocessFn)
    -- Get all the parameters (and gradients) of the model in a single vector.
    local params, gradParams = model:getParameters() -- return 2 tensors

    local options = options or {}
    local batchSize = options.batchSize or 64  -- The bigger the batch size the most accurate the gradients.
    local learningRate = options.learningRate or 0.001  -- This is the learning rate parameter often referred to as lambda.
    local momentumRate = options.momentumRate or 0.9
    local numEpochs = options.numEpochs or 100
    local velocityParams = torch.zeros(gradParams:size())
    local train_features, val_features, test_features

    train_features = trainData.normdata
    val_features = valData.normdata
    test_features = testData.normdata

    preprocessFn = false

    local val_acc_list = {}

    -- Go over the training data this number of times.
    for epoch = 1, numEpochs do
	if epoch % 10 ==0 and learningRate>=0.00001 then 
	    learningRate = learningRate * 0.9
	print ("learningRate is ".. learningRate .. "after" .. epoch .. "epochs") 
	end
        local sum_loss = 0
        local correct = 0
        
        -- Run over the training set samples.
        model:training() -- turn on the training mode
        local n_batches = math.floor(trainData.normdata:size(1) / batchSize)
        for i = 1, n_batches do
            
            -- 1. Sample a batch.
            local inputs
            if preprocessFn then
                inputs = torch.Tensor(batchSize, 3, 224, 224)
            else
                inputs = torch.Tensor(batchSize, 3, 32, 32)
            end
            local labels = torch.Tensor(batchSize)
            for bi = 1, batchSize do
                local rand_id = torch.random(1, train_features:size(1))
                if preprocessFn then
                    -- inputs[bi] = preprocessFn(train_features[rand_id])
                else
                    inputs[bi] = train_features[rand_id]
                end
                labels[bi] = trainData.labels[rand_id]
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
            velocityParams:mul(momentumRate)
            velocityParams:add(learningRate, gradParams)
            params:add(-1, velocityParams)

            if i % n_batches == 0 then  -- Print this every five thousand iterations.
                print(('train epoch=%d, iteration=%d, avg-loss=%.6f, avg-accuracy = %.4f')
                    :format(epoch, i, sum_loss / i, correct / (i * batchSize)))
            end
            xlua.progress(i,n_batches)
        end
        if epoch % 100==0 then
        	torch.save("model/DCNN-epoch" ..  epoch ..".t7", model)
        end


        -- Run over the validation set for evaluation.
        local validation_accuracy = 0
        local nBatches = val_features:size(1) / batchSize
        model:evaluate() -- turn on the evaluation mode
        for i = 1, nBatches do
            
            -- 1. Sample a batch.
            if preprocessFn then
                inputs = torch.Tensor(batchSize, 3, 224, 224)
            else
                inputs = torch.Tensor(batchSize, 3, 32, 32)
            end
            local labels = torch.Tensor(batchSize)
            for bi = 1, batchSize do
                local rand_id = torch.random(1, val_features:size(1))
                if preprocessFn then
                    -- inputs[bi] = preprocessFn(val_features[rand_id])
                else
                    inputs[bi] = val_features[rand_id]
                end
                labels[bi] = valData.labels[rand_id]
            end

            -- 2. Perform the forward pass (prediction mode).
            local predictions = model:forward(inputs)
            
            -- 3. evaluate results.
            for i = 1, predictions:size(1) do
                local _, predicted_label = predictions[i]:max(1)
                if predicted_label[1] == labels[i] then validation_accuracy = validation_accuracy + 1 end
            end
        end
        validation_accuracy = validation_accuracy / (nBatches * batchSize)
        print(('\nvalidation accuracy at epoch = %d is %.4f'):format(epoch, validation_accuracy))

        table.insert(val_acc_list,validation_accuracy)

    end -- for epoch = 1, numEpochs do

    print ("------------------------------------------test time------------------------------------------")
    local test_accuracy = 0
    model:evaluate() -- turn on the evaluation mode
    local inputs = test_features
    local labels = testData.labels
        
    local predictions = model:forward(inputs)
    
    -- 3. evaluate results.
    for i = 1, predictions:size(1) do
        local _, predicted_label = predictions[i]:max(1)
        if predicted_label[1] == labels[i] then test_accuracy = test_accuracy + 1 end
    end

    test_accuracy = test_accuracy / test_features:size(1)
    print(('\ntest accuracy is %.4f'):format(test_accuracy))

    local results = {}
    results.test_accuracy = test_accuracy
    results.val_acc_list = val_acc_list
    torch.save("output/DCNN_result.t7",results)
end

function main()
    trainData, valData, testData = load_data()
    print (trainData, valData, testData)
    trainModel(model,nil, trainData, valData, testData)
end

main()
