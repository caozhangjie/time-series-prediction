require 'torch'
require 'cutorch'
require 'nn'
require 'cunn'
require 'cudnn'

model = nn.Sequential()
model:add(cudnn.LSTM(14, 10, 1, true))
model:add(nn.SplitTable(2))
c = nn.ParallelTable()
for i = 1, 20, 1
do
    c:add(nn.Sequential():add(nn.Linear(10, 1)):add(cudnn.Sigmoid()))
end
model:add(c)
model:add(nn.SelectTable(20))
criterion = nn.BCECriterion()
model:reset(0.001)
if opt.cuda then
    model:cuda()
    criterion:cuda()
end

