data = load('test_13164.mat')
[x,y] = size(data.res)
annos = data.res

fnames = {};
vid = []
cid = []
for i=1:y
    fnames{i} = annos(i).fname;
    vid = [vid annos(i).vid];
    cid = [cid annos(i).class];
end

scale = 0.25;
test_scale = scale / (1 - scale)

train_name = {};
train_size = 1;
train_vid = [];
train_cid = [];
test_name = {};
test_vid = [];
test_cid = [];
test_size = 1;
for label=1:length(unique(cid))
    cate = find(cid==label);
    half_train = int32(length(cate)*scale);
    
    train = cate(randperm(length(cate),half_train));  %当前类下，抽取的训练集的所在行
    test = setdiff(cate,train);   % 当前类下，剩余的也就是测试集的所在行
    half_test = int32(length(test)*test_scale);
    test = test(randperm(length(test),half_test));
    train_len = length(train);
    test_len = length(test);
    for j=1:train_len
        id = train(j);
        train_name{train_size} = fnames{id};
        train_size = train_size + 1;
        train_vid = [train_vid vid(id)];
        train_cid = [train_cid cid(id)];
    end
    
    for k=1:test_len
        idx = test(k);
        test_name{test_size} = fnames{idx};
        test_size = test_size + 1;
        test_vid = [test_vid vid(idx)];
        test_cid = [test_cid cid(idx)];
    end
end

annotations = []
for m=1:train_size-1
    temp = struct('fname',train_name{m},'vid',train_vid(m),'class',train_cid(m));
    annotations = [annotations temp];
end
annotations = annotations(randperm(length(annotations)))
save vehicle_train annotations

annotations = []
for m=1:test_size-1
    temp = struct('fname',test_name{m},'vid',test_vid(m),'class',test_cid(m));
    annotations = [annotations temp];
end
annotations = annotations(randperm(length(annotations)))
save vehicle_test annotations