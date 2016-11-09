local Classifier = torch.class('Classifier')
local xlua = require "xlua"
local util = require "util"
local nn = require 'nn'


-- options contains
function Classifier:__init(options,nInputDim)
    assert (options.model == 'Linear' or options.model == 'CNN',"options.model should be either Linear or CNN")
    self.net = nn.Sequential()
    self.modelname = options.model
    if options.model == 'Linear' then
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


-- function Classifier:evaluate(testData)

-- end


-- function Classifier:getParameters()

-- end


