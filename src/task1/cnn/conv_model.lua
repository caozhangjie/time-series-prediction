require 'torch'
require 'cutorch'
require 'nn'
require 'cunn'
require 'cudnn'

local Convolution
local Pooling
local Activation

if opt.cuda then 
Convolution = cudnn.SpatialConvolution
Pooling = cudnn.SpatialMaxPooling
Activation = cudnn.ReLU
else
Convolution = nn.SpatialConvolution
Pooling = nn.SpatialMaxPooling
Activation = nn.ReLU
end


model = nn.Sequential()
model:add(Convolution(14, 28, 1, 3, 1, 1, 0, 1))
model:add(Activation())
model:add(Pooling(1, 2, 1, 2, 0, 1))
model:add(nn.Dropout(0.5))
model:add(Convolution(28, 56, 1, 3, 1, 1, 0, 1))
model:add(Activation())
model:add(Pooling(1, 2, 1, 2, 0, 0))
model:add(nn.View(-1, 56*5))
model:add(nn.Linear(56*5, 1))
model:add(nn.Sigmoid())
criterion = nn.BCECriterion()
model:reset(0.003)
if opt.cuda then
    model:cuda()
    criterion:cuda()
end

