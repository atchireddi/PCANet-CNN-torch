local xlua = require "xlua"
local util = require "util"
local nn = require 'nn'


function train_classifier(opt, train_features, val_features, test_features, labels)
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

    print(test_features:size())
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
    local opt = opt or {}
    local batchSize = opt.batchSize or 64  -- The bigger the batch size the most accurate the gradients.
    local learningRate = opt.learningRate or 0.001  -- This is the learning rate parameter often referred to as lambda.
    local momentumRate = opt.momentumRate or 0.9
    local numEpochs = opt.numEpochs or 500
    local velocityParams = torch.zeros(gradParams:size())

    preprocessFn = false

    -- Go over the training data this number of times.
    for epoch = 1, numEpochs do
	if epoch%10 ==0 and learningRate>=0.00001 then 
	    learningRate = learningRate * 0.9
	print ("learningRate is ".. learningRate .. "after" .. epoch .. "epochs") 
	end
        local sum_loss = 0
        local correct = 0
        
        -- Run over the training set samples.
        model:training() -- turn on the training mode
        local n_batches = train_features:size(1) / batchSize
        for i = 1, n_batches do
            
            -- 1. Sample a batch.
            local inputs = torch.Tensor(batchSize, p)
            local labels = torch.Tensor(batchSize)
            for bi = 1, batchSize do
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
            velocityParams:mul(momentumRate)
            velocityParams:add(learningRate, gradParams)
            params:add(-1, velocityParams)

            if i % 700 == 0 then  -- Print this every five thousand iterations.
                print(('train epoch=%d, iteration=%d, avg-loss=%.4f, avg-accuracy = %.2f')
                    :format(epoch, i, sum_loss / i, correct / (i * batchSize)))
            end
            xlua.progress(i,n_batches)
        end
        if epoch % 100==0 then
        	torch.save("model/Linear.t7", model)
        end


        -- Run over the validation set for evaluation.
        local validation_accuracy = 0
        local nBatches = val_features:size(1) / batchSize
        model:evaluate() -- turn on the evaluation mode
        for i = 1, nBatches do
            
            -- 1. Sample a batch.
            if preprocessFn then
                -- inputs = torch.Tensor(batchSize, 3, 224, 224)
            else
                inputs = torch.Tensor(batchSize, p)
            end
            local labels = torch.Tensor(batchSize)
            for bi = 1, batchSize do
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
        validation_accuracy = validation_accuracy / (nBatches * batchSize)
        print(('\nvalidation accuracy at epoch = %d is %.4f'):format(epoch, validation_accuracy))
    end -- for epoch = 1, numEpochs do

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

end


function main()
    print ("loading features")
    local timer = torch.Timer() 
    timer:reset()
	train_features = torch.load("features/train_features.t7")
	val_features = torch.load("features/val_features.t7")
	test_features = torch.load("features/test_features.t7")
	labels = torch.load("features/labels.t7")
    print('Time elapsed for loading features: ' .. timer:time().real .. ' seconds')

    
	train_classifier(nil,train_features,val_features,test_features, labels)
end



main()










