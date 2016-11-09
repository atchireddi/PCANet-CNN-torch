local Classifier = torch.class('Classifier')
local xlua = require "xlua"
local util = require "util"
local nn = require 'nn'


-- options contains
function Classifier:__init(options,nInputDim)
    assert (options.model == 'SVM' or options.model == 'CNN',"options.model should be either SVM or CNN")
    self.net = nn.Sequential()
    self.modelname = options.model
    if options.model == 'SVM' then
        self.net:add(nn.Linear(nInputDim, 10)):add(nn.LogSoftMax()) -- what's the dimension of nInput
    elseif options.model == 'CNN' then
        self.net:add(nn.SpatialConvolution(nInputDim, 8, 5, 5))  -- 3 input channels, 8 output channels (8 filters), 5x5 kernels.
        self.net:add(nn.SpatialBatchNormalization(8, 1e-3))  -- BATCH NORMALIZATION LAYER.
        self.net:add(nn.ReLU())
        self.net:add(nn.SpatialMaxPooling(2, 2, 2, 2)) -- Max pooling in 2 x 2 area.
        self.net:add(nn.SpatialConvolution(8, 16, 5, 5))  -- 8 input channels, 16 output channels (16 filters), 5x5 kernels.
        self.net:add(nn.SpatialBatchNormalization(16, 1e-3))  -- BATCH NORMALIZATION LAYER.
        self.net:add(nn.ReLU())
        self.net:add(nn.SpatialMaxPooling(2, 2, 2, 2))  -- Max pooling in 2 x 2 area.
        self.net:add(nn.View(16*5*5))    -- Vectorize the output of the convolutional layers.
        self.net:add(nn.Linear(16*5*5, 120))
        self.net:add(nn.ReLU())
        self.net:add(nn.Linear(120, 84))
        self.net:add(nn.ReLU())
        self.net:add(nn.Linear(84, 10))
        self.net:add(nn.LogSoftMax())
    end
    -- Negative log-likelihood criterion.
    self.criterion = nn.ClassNLLCriterion() 
end



-- function Classifier:train(trainData)
-- 	-- 2. training NN
-- 	-- if paths.filep("NN.t7") then 
-- 	timer:reset()
-- 	print ("training...")

-- 	-- 2. Select a batch of data and pass it through PCA filter first, then linear layer 
-- 	print ("start momentum SGD")
-- 	-- Go over the training data this number of times.
-- 	local params, gradParams = model:getParameters() -- return 2 tensors
-- 	local velocityParams = torch.zeros(gradParams:size())
	
-- 	for epoch = 1, options.max_epochs do
--         local sum_loss = 0
--         local correct = 0
        
--         -- Run over the training set samples.
--         model:training() -- turn on the training mode
--         local n_batches = math.floor(trainData.data:size(1) / options.batch_size)
--         for i = 1, n_batches do
--             -- print ("1.1 Sample a batch")
--             local img_batch_i = torch.Tensor(options.batch_size, 3, 32, 32)
--             local labels = torch.Tensor(options.batch_size)

--             for bi = 1, options.batch_size do
--                 local rand_id = torch.random(1, trainData.data:size(1))
--                 img_batch_i[bi] = trainData.data[i]
--                 labels[bi] = trainData.labels[rand_id]
--             end
--             inputs = pcanet:PCANet_FeaExt(img_batch_i)
--             assert(inputs:size(2)==nInputDim,"dim not match")

--             -- print ("1.2 Perform the forward pass (prediction mode).")
--             local predictions = model:forward(inputs)
            
--             -- print("1.3 Evaluate results")
--             for i = 1, predictions:size(1) do
--                 local _, predicted_label = predictions[i]:max(1)
--                 if predicted_label[1] == labels[i] then correct = correct + 1 end
--             end
--             sum_loss = sum_loss + criterion:forward(predictions, labels)

--             -- print("1.4 Perform the backward pass (compute derivatives)")
--             -- This zeroes-out all the parameters inside the model pointed by variable params.
--             model:zeroGradParameters()
--             -- This internally computes the gradients with respect to the parameters pointed by gradParams.
--             local gradPredictions = criterion:backward(predictions, labels)
--             model:backward(inputs, gradPredictions)

--             -- print ("1.5 Momentum update")
-- 			-- v = mu * v - options.learning_rate * dx -- integrate velocity
-- 			-- x += v -- integrate position
--             velocityParams:mul(options.momentum_rate)  
--             velocityParams:add(options.learning_rate, gradParams)
--             params:add(-1, velocityParams)

--             xlua.progress(i,n_batches)
--             if i % 100 == 0 then  
--                 print(('train epoch=%d, iteration=%d, avg-loss=%.6f, avg-accuracy = %.2f')
--                     :format(epoch, i, sum_loss / i, correct / (i * options.batch_size)))
--             end
--         end

        
--         -- print("after each epoch, evaluate the accuracy for the val data")
--         local validation_accuracy = 0
--         model:evaluate() -- turn on the evaluation mode

--         -- Perform the forward pass (prediction mode).
--         local predictions = model:forward(pcanet:PCANet_FeaExt(ValData.data))
--         -- evaluate results.
--         for i = 1, predictions:size(1) do
--             local _, predicted_label = predictions[i]:max(1)
--             if predicted_label[1] == ValData.labels[i] then validation_accuracy = validation_accuracy + 1 end
--         end

--         validation_accuracy = validation_accuracy / (ValData.data:size(1))
--         print(('\n validation accuracy at epoch = %d is %.4f'):format(epoch, validation_accuracy))


--         print("learning rate decay")
--         options.learning_rate = options.learning_rate * options.learning_rate_decay
--         options.learning_rate = math.max(options.learning_rate,0.0001)

--         if epoch % 5==0 then
--         	torch.save("model.t7",model)
--         end
--         collectgarbage()

--     end

--     print('Time elapsed for training NN: ' .. timer:time().real .. ' seconds')


--     print ("test time")
--     local test_accuracy = 0
--     model:evaluate() -- turn on the evaluation mode

--     -- Perform the forward pass (prediction mode).
--     local predictions = model:forward(pcanet:PCANet_FeaExt(testData.data))
--     -- evaluate results.
--     for i = 1, predictions:size(1) do
--         local _, predicted_label = predictions[i]:max(1)
--         if predicted_label[1] == testData.labels[i] then test_accuracy = test_accuracy + 1 end
--     end

--     test_accuracy = test_accuracy / (testData.data:size(1))
--     print(test_accuracy)
-- end


-- function Classifier:evaluate(testData)

-- end


-- function Classifier:getParameters()

-- end


