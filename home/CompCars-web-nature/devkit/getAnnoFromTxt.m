f = fopen('train.txt')

tline = fgetl(f);
list_cell = {};
i = 1

while ischar(tline)
    disp(tline);
    tline = fgetl(f);
    list_cell{i} = tline;
    i = i + 1;
end
fclose(f);


len = length(list_cell);

annotations = [];
for i=1:len
    path_name = list_cell{i};
    if path_name == -1
        break;
    end
    attrs = strsplit(path_name,'/');
    model_id = attrs{2};
    item = struct('fname',path_name,'model',str2num(model_id));
    annotations = [annotations item];
end
annotations = annotations(randperm(length(annotations)));
save compcars_train annotations