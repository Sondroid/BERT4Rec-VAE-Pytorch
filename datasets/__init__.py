from .yna import YNADataset

DATASETS = {
    YNADataset.code(): YNADataset
}


def dataset_factory(args):
    dataset = DATASETS[args.dataset_code]
    return dataset(args)
