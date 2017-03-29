function purity = cPurity(label, truth, class_num)
    purity = 0;
    all_length = length(label);
    for i = 1:1:class_num
        select_data = label == i;
        select_label = truth(select_data);
        [~, temp_purity] = mode(select_label);
        temp_purity = temp_purity / length(select_label);
        purity = purity + temp_purity * length(select_label) / all_length;
    end
end