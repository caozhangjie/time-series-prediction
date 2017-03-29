require 'torch'
require 'nn'

if opt.cuda then
require 'cutorch'
require 'cunn'
require 'cudnn'
end

model = nn.Sequential()
model:add(nn.View(-1, 20))
model:add(nn.Linear(20, 10))
model:add(nn.ReLU())
model:add(nn.Linear(10, 1))
model:add(nn.Sigmoid())
criterion = nn.BCECriterion()

if opt.cuda then
    model:cuda()
    criterion:cuda()
end

