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




function train(options, trainData, testData, preprocess)
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


    -- extract the features of train, val, and test data

    print ("extracting train features")
    timer:reset()
    local train_features = pcanet:PCANet_FeaExt(trainData.data)
    torch.save("features/train_features.t7",train_features)
    train_features = nil
    collectgarbage()
    print('Time elapsed for training features: ' .. timer:time().real .. ' seconds')


    print ("extracting val features")
    timer:reset()
    local val_features = pcanet:PCANet_FeaExt(valData.data)
    torch.save("features/val_features.t7",val_features)
    val_features = nil
    collectgarbage()
    print('Time elapsed for val features: ' .. timer:time().real .. ' seconds')

    print ("extracting test features")
    timer:reset()
    local test_features = pcanet:PCANet_FeaExt(testData.data)
    torch.save("features/test_features.t7",test_features)
    test_features = nil
    collectgarbage()
    print('Time elapsed for test features: ' .. timer:time().real .. ' seconds')

    -- output the labels
    labels = {trainData.labels, valData.labels, testData.labels}
    torch.save("features/labels.t7",labels)

end




function main(options)
    trainData, valData, testData = load_data()
    -- print (trainData)
    train(options, trainData, valData, testData)
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


-- model
cmd:option('-model','SVM', 'either SVM or CNN')


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






















