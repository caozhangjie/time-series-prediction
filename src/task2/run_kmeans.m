load ./data
truth = label;
test_k = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20];
length_of_test = length(test_k);
purity = zeros(1,length_of_test);
f_score = zeros(1,length_of_test);
iter_num = zeros(1,length_of_test);
all_label = zeros(length(label), length_of_test);
all_loss = zeros(1, length_of_test);
all_center = zeros(length_of_test*(length_of_test-1)/2, 2);
for i = 1:1:length_of_test
    [center, label, loss, iter_num(i)] = Kmeans(test_k(i), data);
    all_label(:, i) = label;
    all_loss(i) = loss;
    all_center((i-1)*i/2+1:i*(i+1)/2, :) = center;
    purity(i) = cPurity(label, truth, i);
    f_score(i) = Fscore(label, truth, i);
end
save result.mat all_label all_loss all_center purity f_score iter_num
