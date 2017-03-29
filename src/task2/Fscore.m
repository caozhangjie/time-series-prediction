function f_score = Fscore(label, truth, class_num)
    right_pair_num = 572633193;
    all_label = [46, 2501, 861, 11543, 93036, 2333, 209, 74683, 2226, 843, 13556];
    f_score = 0;
    true_pair_num = 0;
    all_pair_num = 0;
    for i = 1:1:class_num
        select_data = label == i;
        select_label = truth(select_data);
        temp = sum(select_data);
        all_pair_num = all_pair_num + temp * (temp-1) / 2;
        for j = 1:1:11
            this_class = select_label == all_label(j);
            temp = sum(this_class);
            true_pair_num = temp*(temp-1) / 2 + true_pair_num;
        end
        precision = true_pair_num / all_pair_num;
        recall = true_pair_num / right_pair_num;
        f_score = 2/(1/precision+1/recall);
    end
end