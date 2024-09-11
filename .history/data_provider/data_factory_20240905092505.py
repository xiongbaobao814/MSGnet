from .data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred, Dataset_Flight
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'Flight':Dataset_Flight,
}


# flag = 'train' or 'val' or 'test'
def data_provider(args, flag):
    Data = data_dict[args.data]
    #time features encoding, options: [timeF, fixed, learned]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred

    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    data_set = Data(root_path=args.root_path,
                    data_path=args.data_path,
                    flag=flag,
                    size=[args.seq_len, args.label_len, args.pred_len],
                    features=args.features,
                    target=args.target,
                    timeenc=timeenc,
                    freq=freq,
                    seasonal_patterns = args.seasonal_patterns
    )
    print(flag, len(data_set))
    # data_x:(34560, 7), data_y:(34560, 7)

    # x = []
    # for i, data in enumerate(data_set):
    #     inputs, labels = data
    #     x.append(inputs)
    # print(len(x))
    
    data_loader = DataLoader(data_set,
                            batch_size=batch_size,
                            shuffle=shuffle_flag,
                            num_workers=args.num_workers,
                            drop_last=drop_last
    )
    return data_set, data_loader