import numpy as np
import matplotlib.pyplot as plt

# Global Configuration
labels_acc = np.array(['Raw Acc', 'Crop Acc', 'Drop Acc', 'Val Acc'])   # static params
legend_label_list = np.array(['GAP', 'GAP-FPN', 'GAP-W', 'GAP-W-FPN'])  # labels
epoch_loss_scope = [10, 50]
epoch_acc_scope = [10, 50]
log_name = ['gap_train.log', 'gap_fpn_train.log', 'gap_w_train.log', 'gap_w_fpn_train.log']  # path name
log_cnt = 4


def get_split_acc(line, label):
    tmp = line.split(label + ' (')
    tmp2 = tmp[1][0:tmp[1].index(')')]
    return tmp2.split(',')


def get_data_list_from_log(log):
    train_loss_list = []
    val_loss_list = []
    raw_acc_list = []
    crop_acc_list = []
    drop_acc_list = []
    val_acc_list = []
    idx = 0
    with open(log, 'r') as f:
        for line in f:
            if line.find('Train: Loss') > -1:
                idx += 1
                train_loss_list.append(line.split('Loss ')[1][0:6])
                raw_acc_list.append(get_split_acc(line, labels_acc[0]))
                crop_acc_list.append(get_split_acc(line, labels_acc[1]))
                drop_acc_list.append(get_split_acc(line, labels_acc[2]))
            if line.find('Valid: Val Loss') > -1:
                val_loss_list.append(line.split('Valid: Val Loss ')[1][0:6])
                val_acc_list.append(get_split_acc(line, labels_acc[3]))
            else:
                continue
    return idx, \
           np.array(train_loss_list).astype('float32'), np.array(val_loss_list).astype('float32'), \
           np.array(raw_acc_list).astype('float32'), np.array(crop_acc_list).astype('float32'), \
           np.array(drop_acc_list).astype('float32'), np.array(val_acc_list).astype('float32')


def pred_sub_target(arr):
    res = []
    for idx in range(arr.shape[0]):
        new_item = arr[idx][0] - arr[idx][1]
        res.append(new_item)
    return np.array(res)


def get_partial_item(gross_arr, part):
    res = []
    for arr in gross_arr:
        sub_res = []
        for idx in range(arr.shape[0]):
            new_item = arr[idx][part]
            sub_res.append(new_item)
        res.append(sub_res)
    return np.array(res, dtype=object)


color_list = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w']


def get_gross_comparison(cnt, train_list, val_list, label_list, legend_list, scope, title):
    x = np.arange(scope[0], scope[1] + 1)  # epoch time
    fig, ax = plt.subplots()
    for idx in range(cnt):  # how many pairs of train & validate data
        ax.plot(x, train_list[idx][scope[0]:scope[1] + 1], color_list[idx], label=legend_list[idx] + '-train')
        ax.plot(x, val_list[idx][scope[0]:scope[1] + 1], color_list[idx] + '--', label=legend_list[idx] + '-val')
        # train's line is solid, val's line is dashed

    ax.set_xlabel(label_list[0])
    ax.set_ylabel(label_list[1])
    ax.set_title(title)
    ax.legend()
    plt.grid(True)
    plt.show()


def get_val_acc_comparison(cnt, acc_list, label_list, legend_list, scope, title):
    x = np.arange(scope[0], scope[1] + 1)  # epoch time
    fig, ax = plt.subplots()
    for idx in range(cnt):  # how many pairs of train & validate data
        ax.plot(x, acc_list[idx][scope[0]:scope[1] + 1], color_list[idx] + '--', label=legend_list[idx])
        # train's line is solid, val's line is dashed

    ax.set_xlabel(label_list[0])
    ax.set_ylabel(label_list[1])
    ax.set_title(title)
    ax.legend()
    plt.grid(True)
    plt.show()


def get_best_entry_from_log(acc_list, loss_list):
    acc = np.array(acc_list)
    loss = np.array(loss_list)
    tmp = acc - loss
    idx = np.argmax(tmp)
    return acc_list[idx], loss_list[idx]


def get_best_acc(acc_lists, loss_lists):
    max_acc = []
    correspond_loss = []
    for i in range(legend_label_list.size):
        max_acc_idx = np.argmax(acc_lists[i])
        max_acc.append(acc_lists[i][max_acc_idx])
        correspond_loss.append(loss_lists[i][max_acc_idx])
        print('[{0:9}] Val loss({2:.4f}) , Val Acc({1:.2f}%)'.format(legend_label_list[i], max_acc[i], correspond_loss[i]))
    return max_acc, correspond_loss


def check_size(scope, size_list, label):
    start = scope[0]
    end = scope[1]
    scope_size = start - end + 1
    if_terminate = False
    print("[Examine] Check {} list' alignment......".format(label))
    for i in range(len(size_list)):
        size_t = size_list[i]
        if start > size_t - 1:
            if_terminate = True
            print("[Error] Scope ValueError, the ({0})'s epoch is {1}, but the scope start from {2}". \
                  format(legend_label_list[i], size_t, start))
        if end > size_t - 1:
            if_terminate = True
            print("[Error] Scope ValueError, the ({0})'s epoch is {1}, but the scope end by {2}". \
                  format(legend_label_list[i], size_t, end))
        if scope_size > size_t:
            if_terminate = True
            print("[Error] Scope ValueError, the ({0})'s epoch is {1}, but the scope requires {2}". \
                  format(legend_label_list[i], size_t, scope_size))
    return if_terminate


def get_statistics():
    print("[Statistic] Get data initialization.....")
    # initialization
    size_list = []
    train_loss_list = []
    val_loss_list = []
    raw_acc_list = []
    crop_acc_list = []
    drop_acc_list = []
    val_acc_list = []

    print("[Statistic] Get the data lists.....")
    # get data
    for idx in range(log_cnt):
        size_l, train_loss_l, val_loss_l, raw_acc_l, crop_acc_l, drop_acc_l, val_acc_l = get_data_list_from_log(
            log_name[idx])
        size_list.append(size_l)
        train_loss_list.append(train_loss_l)
        val_loss_list.append(val_loss_l)
        raw_acc_list.append(raw_acc_l)
        crop_acc_list.append(crop_acc_l)
        drop_acc_list.append(drop_acc_l)
        val_acc_list.append(val_acc_l)
        print("[Statistic] Successfully get the data from '{0}' , training epochs :{1}".format(log_name[idx], size_l))

    print("[Statistic] Re-format data.....")
    # re-format data
    size_list = np.array(size_list)
    train_loss_list = np.array(train_loss_list)
    val_loss_list = np.array(val_loss_list)
    raw_acc_list = np.array(raw_acc_list)
    crop_acc_list = np.array(crop_acc_list)
    drop_acc_list = np.array(drop_acc_list)
    val_acc_list = np.array(val_acc_list)

    gross_train_loss_list = train_loss_list
    gross_val_loss_list = val_loss_list
    gross_train_acc_list = get_partial_item(raw_acc_list, 0)
    gross_val_acc_list = get_partial_item(val_acc_list, 0)

    print("[Statistic] Proofread the scope and data size.....")
    check_size(epoch_acc_scope, size_list, 'Accuracy')
    check_size(epoch_loss_scope, size_list, 'loss')

    print("[Statistic] Getting the result ...")
    # draw statistics
    get_gross_comparison(log_cnt, gross_train_loss_list, gross_val_loss_list, ['Epoch', 'Loss'], legend_label_list,
                         epoch_loss_scope,
                         'Loss Comparison')
    get_gross_comparison(log_cnt, gross_train_acc_list, gross_val_acc_list, ['Epoch', 'Precision(%)'],
                         legend_label_list,
                         epoch_acc_scope,
                         'Accuracy Comparison')
    get_val_acc_comparison(log_cnt, gross_val_acc_list, ['Epoch', 'Precision(%)'], legend_label_list,
                           epoch_acc_scope, 'Validated Accuracy')
    # print(legend_label_list)
    print("[Statistic] Get the best entry with best accuracy...")
    get_best_acc(gross_val_acc_list, gross_val_loss_list)
    print("[Statistic] Statistic Finished!")


def main():
    get_statistics()

main()