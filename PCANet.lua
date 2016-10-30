local PCANet = torch.class('PCANet')
local xlua = require "xlua"
local list = require "pl.List"
local util = require "util"

function PCANet:__init(options)
    options = options or {}
    -- patches
    self.kW = options.kW or 7
    self.kH = options.kH or 7
    self.dW = options.dW or 1
    self.dH = options.dH or 1
    self.padW = options.padW or 0
    self.padH = options.padH or 0

    -- PCA filter
    self.NumFilters = options.NumFilters or {8,8}
    self.NumStages = #self.NumFilters
    self.V = {}

    -- binary hashing and block histogram
    self.HistBlockSize = options.HistBlockSize or {8,6}
    self.BlkOverLapRatio = options.BlkOverLapRatio or 0.5
    assert(self.BlkOverLapRatio<1 and self.BlkOverLapRatio>=0,"self.BlkOverLapRatio must be in [0,1)")
end


--  Randomly select a MaxSamples patches, and extract the first NumFilter PCA filters
function PCANet:PCA_FilterBank(InImgs,stage, MaxSamples)
    --[[ 
      InImgs must be a 4D tensor N,C,W,H
    ]]
    assert(InImgs:dim()==4,"InImgs must be a 4D tensor (N,C,W,H)")

    local N = InImgs:size(1)
    local C = InImgs:size(2)
    local MaxSamples = MaxSamples or 100000
    local NumRSamples = math.min(N, MaxSamples)

    local xxT = torch.Tensor(self.kW*self.kH*C,self.kW*self.kH*C):zero()  
    --patch size is self.kW*self.kH*C

    for i = 1, NumRSamples do
        local r = math.ceil(torch.rand(1)[1] * N)
        xi = util.im2col_mean_removal(InImgs[r],self.kW,self.kH,self.dW,self.dH,self.padW,self.padH,true)
        xxT = xxT + torch.mm(xi,xi:t())
    end

    xxT = xxT / (NumRSamples * xi:size(2)) -- number of columns is NumRSamples * xi:size(2)

    local e, V  = torch.eig(xxT,'V')
    V = V[{ {1,self.NumFilters[stage]},{}}]

    return V
end

--  Caluclate the PCA score of each patch in a single stage, Given the PCA_Filters 
function PCANet:PCA_output(InImgs,stage)
    --[[ 
      InImgs must be a 4D tensor N,C,W,H
    ]]
    assert(InImgs:dim()==4,"InImgs must be a 4D tensor (N,C,W,H) ")

    local N = InImgs:size(1)
    local C = InImgs:size(2)
    local W = InImgs:size(3)
    local H = InImgs:size(4)

    -- the dimension of out image (N*L_1,oW,oH)
    local oH = math.floor((H + 2 * self.padH - self.kH) / self.dH) + 1  -- Height of output
    local oW = math.floor((W + 2 * self.padW - self.kW) / self.dW) + 1 -- Width of output

    outImgs = nil
    for i=1,N do
        local img = InImgs[i]
        local xi = util.im2col_mean_removal(img,self.kW,self.kH,self.dW,self.dH,self.padW,self.padH,true)
        -- just simple matrix multiplication to get the PCA scores
        local score = torch.mm(self.V[stage],xi):reshape(self.NumFilters[stage],1,oW,oH)  
        --TODO: some functions on score

        --        
        if not outImgs then
            outImgs = score
        else
            outImgs = torch.cat({outImgs,score},1)
        end
    end
    return outImgs -- the dimension of out image (N*self.NumFilters[stage],1,oW,oH)
end


-- to obtain all the PCA filters for all the stages
function PCANet:PCANet_train(InImgs,MaxSamples)
	assert(InImgs:dim()==4,"InImgs must be a 4D tensor (N,C,W,H) ")
    local MaxSamples = MaxSamples or 100000

    local N = InImgs:size(1)
    local OutImgs = InImgs:clone()
    self.V = {} -- when training the net, reset the filters
    for stage = 1,self.NumStages do
        print(string.format("Computing PCA filter bank and its outputs at stage %d ...", stage))
        local tmpV = self:PCA_FilterBank(OutImgs, stage, MaxSamples)
        table.insert(self.V,tmpV)

        if stage~=self.NumStages then
            OutImgs= self:PCA_output(OutImgs,stage)
        end
    end
    -- return self.V
    print ("PCA filter bank training completed!")
end

function PCANet:PCA_FilterBank_viz()

end

-- only applied to n_stage = 2
function PCANet:HashingHist(OutImg) 
	-- dimension of OutImg is (L1*L2,1,oW,oH)
	-- OutImg is a single image
    local L_2 = self.NumFilters[#self.NumFilters] -- L_2 is the number of PCA_filters in last stage
    local L_1 = OutImg:size(1)/L_2   -- L_2 is the number of PCA_filters in last stage
    local W = OutImg:size(3)
    local H = OutImg:size(4)
    
    -- the size of output is (L_1, OutImg:size(2),OutImg:size(3),OutImg:size(4))
    local output = torch.Tensor(L_1, OutImg:size(2),OutImg:size(3),OutImg:size(4)):zero()
    for i= 1, L_1 do 
        local t = OutImg[{  { (i-1)*L_2+1,i*L_2   }, {},{},{}    }]:clone()
        t:apply(function(x) if x>0 then return 1 else return 0 end end  ) -- binarization hashing
        local pow = L_2 - 1
        for j=1,t:size(1) do
            t[j] = t[j] * 2^pow
            output[i] = output[i] + t[j]
            pow = pow - 1
        end
    end

    local histsize = 2^L_2
    local skipW = (1-self.BlkOverLapRatio)*self.HistBlockSize[1] --skipW
    local skipH = (1-self.BlkOverLapRatio)*self.HistBlockSize[2] --skipH
    local m = math.floor(W/skipW)
    local n = math.floor(H/skipH)
    
    -- set up the block size (B = m x n) and calculate the histogram, bin size is 2^L_2
    -- f is the output feature 
    -- the feature f will have length of  if n_stage = 2

    local f = nil  -- initalization
    -- print (m,n)
    for l=1, L_1 do  
        for i=1, m do
            for j = 1,n do 
                -- print (i,j)
                local starti = (i-1)*skipW+1; local startj = (j-1)*skipH+1;
                local endi = starti - 1 + self.HistBlockSize[1]; 
                local endj = startj - 1 + self.HistBlockSize[2];
                if i==m then endi = W end 
                if j==n then endj = H end

                local patch_ij = output[{  {l},{},{ starti,endi  }  ,{startj,endj}  }]:clone() -- the ij th patch
                local f_ij = torch.histc(patch_ij, histsize, 0, histsize-1)      -- the feature of ij the patch
                if not f then
                    f = f_ij
                else
                    f = torch.cat({f,f_ij},1)
                end
            end
        end
    end

    return f 
end

-- --  process a single image, transform raw image to final representation f, given the PCAFilter V and Blk_size
-- function PCANet:PCANet_FeaExt(PCANet, InImg, V, Blk_size)
function PCANet:PCANet_FeaExt_single_input(InImg)
	assert(#self.V==self.NumStages,"PCANet must be trained first")
	-- InImg has size (C,W,H)
	if InImg:dim()==3 then
		InImg:resize(1,InImg:size(1),InImg:size(2),InImg:size(3))
	end
	for stage = 1, #self.V do
		InImg = self:PCA_output(InImg,stage) --input must be a 4D tensor N,C,W,H
	end
    local f = self:HashingHist(InImg)
    return f
end

function PCANet:PCANet_FeaExt(InImgs)
	assert(#self.V==self.NumStages,"PCANet must be trained first")
	assert(InImgs:dim()==4,"InImgs must be a 4D tensor (N,C,W,H) ")
	local features = nil
	for i = 1,InImgs:size(1) do
		local f = self:PCANet_FeaExt_single_input(InImgs[i])
        f:resize(1,f:size(1))
        if not features then
            features = f
		else 
            features = torch.cat({features,f},1)
        end
	end
	return features
end

