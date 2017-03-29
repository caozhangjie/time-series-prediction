function [center, label, loss, iter_num] = Kmeans(k, data)
    n = size(data, 1);
    dimension = size(data, 2);
    initialized_center = int32(rand(1, k) * n);
    center = data(initialized_center, :);
    label = zeros(1,n);
    new_label = zeros(1,n);
    loss = 0;
    sum_of_class = zeros(k, dimension);
    class_size = zeros(k ,1);
    same_num = 0;
    iter_num = 0;
    while true
        iter_num = iter_num + 1;
        iter_num
        for i = 1:1:n
            temp = repmat(data(i, :), k, 1);
            diff = sum(power(temp - center, 2.0), 2);
            [~, new_label(i)] = min(diff);
            loss = loss + sqrt(diff(new_label(i)));
            sum_of_class(new_label(i), :) = sum_of_class(new_label(i), :) + data(i, :);
            class_size(new_label(i), 1) = class_size(new_label(i), 1) + 1;
            if new_label(i) == label(i)
                same_num = same_num + 1;
            end
        end
        for i = 1:1:n
            label(i) = new_label(i);
        end
        center = sum_of_class ./ repmat(class_size, 1, dimension);
        sum_of_class = zeros(k, dimension);
        class_size = zeros(k ,1);
        if same_num < n
            same_num = 0;
            loss = 0;
            continue
        else
            break
        end
    end
end
