from data.data_loader import Dataset_Custom, Dataset_Pred, MultiTimeSeries
from torch.utils.data import DataLoader

data_dict = {
    'custom': Dataset_Custom,
    'covid': MultiTimeSeries
}


def data_provider(args, flag):
    timeenc = 0 if args.embed != 'timeF' else 1
    if flag == 'pred':
        Data = Dataset_Pred
        batch_size = 1
    else:
        Data = data_dict[args.data]
        batch_size = args.batch_size
    
    drop_last = flag == 'train'
    shuffle_flag = flag == 'train'
    freq = args.freq

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        scale=not args.no_scale
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
