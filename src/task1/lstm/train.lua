require 'torch'
require 'xlua'
require 'optim'
require 'gnuplot'
require 'pl'
require 'trepl'
require 'nn'
----------------------------------------------------------------------

cmd = torch.CmdLine()
cmd:addTime()
cmd:text()
cmd:text('Training a convolutional network for visual classification')
cmd:text()
cmd:text('==>Options')

cmd:text('===>Model And Training Regime')
cmd:option('-LR',                 0.003,                   'learning rate')
cmd:option('-LRDecay',            0,                      'learning rate decay (in # samples)')
cmd:option('-weightDecay',        1e-5,                    'L2 penalty on the weights')
cmd:option('-momentum',           0.9,                    'momentum')
cmd:option('-batch_size',          30,                    'batch size')
cmd:option('-test_batch_size',     1,               'test batch size')
cmd:option('-epoch_num',              100,                     'number of epochs to train, -1 for unbounded')

cmd:text('===>Platform Optimization')
cmd:option('-cuda', true, 'whether use cuda or not')
cmd:option('-devid',              1,                      'device ID (if using CUDA)')


cmd:text('===>Save/Load Options')
cmd:option('-load',               '',                     'load existing net weights')
cmd:option('-save',               "lstm", 'save directory')

torch.manualSeed(432)
opt = cmd:parse(arg or {})
opt.save = paths.concat('../Results', opt.save)
os.execute("mkdir -p ".. opt.save)

if opt.cuda then
    require 'cutorch'
    require 'cunn'
    require 'cudnn'
end

dofile("./lstm_model.lua")

npyio = require "npy4th"
data0 = npyio.loadnpy("../data/mean/n_train_data.npy")

size_data = data0:size()
num_data0 = data0:size(1)
size_data[1] = num_data0
data_random = torch.Tensor(data0:size()):zero()
random_order = torch.randperm(size_data[1])
for i = 1, size_data[1], 1
do
    data_random[{{i}}] = data0[{{random_order[i]}}]
end

test_data0 = npyio.loadnpy("../data/mean/n_test_data.npy")
test_size_data = test_data0:size()
test_num_data0 = test_data0:size(1)
test_size_data[1] = test_num_data0
test_data_all = test_data0


local Weights, Gradients = model:getParameters()
optimState = {
    learningRate = opt.LR,
    momentum = opt.momentum,
    weightDecay = opt.weightDecay,
    learningRateDecay = opt.LRDecay
}
optimMethod = optim.sgd

all_labels = npyio.loadnpy("../data/mean/test_label.npy")
train_all_labels = npyio.loadnpy("../data/mean/train_label.npy")


for i =1, opt.epoch_num,1
do
    model:training()
    train_loss = 0
    batch_num = torch.floor((num_data0) / opt.batch_size)
    order_epoch = torch.randperm(batch_num)
    for j = 1, batch_num, 1
    do
        train_data = data_random[{{(order_epoch[j]-1) * opt.batch_size+1, order_epoch[j] * opt.batch_size}, {},{}}]:transpose(2,3)
        train_label = train_all_labels[{{(order_epoch[j]-1) * opt.batch_size+1, order_epoch[j] * opt.batch_size}}]:reshape(opt.batch_size, 1)
        if opt.cuda then
            train_data = train_data:cuda()
            train_label = train_label:cuda()
        end
        output = model:forward(train_data)
        curloss = criterion:forward(output, train_label)
        function feval()
            model:zeroGradParameters()
            local dE_dy = criterion:backward(output, train_label)
            model:backward(train_data, dE_dy)
            return curloss, Gradients
        end
        optimMethod(feval, Weights, optimState)
        train_loss = train_loss + curloss
    end
    print("Training Loss: " .. tostring(train_loss))

    model:evaluate()
    all_output = torch.Tensor(test_num_data0, 1)
    test_loss = 0
    test_batch_num = torch.floor((test_num_data0) / opt.test_batch_size)
    order_epoch = torch.randperm(test_batch_num)
    for j = 1, test_batch_num, 1
    do
        test_data = test_data_all[{{(order_epoch[j]-1) * opt.test_batch_size+1, order_epoch[j] * opt.test_batch_size}, {}, {}}]:transpose(2,3)
        test_label = all_labels[{{(order_epoch[j]-1) * opt.test_batch_size+1, order_epoch[j] * opt.test_batch_size}}]:reshape(opt.test_batch_size)
        if opt.cuda then
            test_data = test_data:cuda()
            test_label = test_label:cuda()
        end
        output = model:forward(test_data)
        curloss = criterion:forward(output, test_label)
        test_loss = test_loss + curloss
        all_output[{{(j-1)*opt.test_batch_size+1,j*opt.test_batch_size}, {1}}]:copy(output)
    end
    print("Testing Loss: " .. tostring(test_loss))
    predict_label = torch.lt(all_output:double(), 0.5)
    diff = torch.sum(torch.abs(predict_label:double() - all_labels:double()))
    print("Test Accuracy: " .. tostring(diff / predict_label:nElement()))
    torch.save(opt.save .. "/model.net", model)
end
