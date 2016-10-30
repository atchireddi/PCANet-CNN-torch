
local util = {}


-- example, input images are 500x3x32x32, 3, 3, 1, 1, 0, 0 => 3500x3x30x30 if number of filters = 7

-- function im2col_mean_removal: vectorize all the patches of an image
function util.im2col_mean_removal(InImg,kW,kH,dW,dH,padW,padH,remove_mean)
	--[[
	src: input image
	kW, KH: patch size
	dW,dH: stride size
	padW,padH: zero padding size, must be 0 at this moment
	remove_mean: substract the patch mean from the patch
	]]
	
	-- dimension order: C,W,H
    if InImg:dim()==2 then 
        InImg:reshape(1,InImg:size(1),InImg:size(2))
    end

    local C = InImg:size(1)
    local W = InImg:size(2)
    local H = InImg:size(3)

    local oH = math.floor((H + 2 * padH - kH) / dH) + 1  -- Height of output
    local oW  = math.floor((W + 2 * padW - kW) / dW) + 1 -- Width of output
    local n = oH * oW
	
	-- output dimension (kW * kH * C, oH * oW )
	local output = torch.Tensor(kW * kH * C, n):zero()  -- must initialized at 0
	local idx = 1
	-- for c = 1, C do
		for w = 1, oW do
			for h = 1, oH do
				tmp = InImg[{  {} , {(w-1) * dW +1, (w-1) * dW + kW} , {(h-1) * dH +1, (h-1) * dH +kH}  }]:clone()
				tmp:resize(tmp:nElement())
				if remove_mean then
					tmp:add(-tmp:mean())
				end
				output[{{},{idx}}] = tmp
				idx = idx + 1
			end
		end
	-- end

	return output
end

return util