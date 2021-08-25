def set_template(args):
    if args.template is None:
        return

    elif args.template.startswith('train_bert'):
        args.mode = 'train'

        args.dataset_code = 'yna'
        args.dataset_datetime = '21061212'

        FULL_BATCH = 999999
        args.train_batch_size = 1024
        args.val_batch_size = FULL_BATCH
        args.test_batch_size = FULL_BATCH

        args.device = 'cuda'
        args.num_gpu = 1
        args.device_idx = '0'
        args.optimizer = 'Adam'
        args.lr = 0.0025
        args.enable_lr_schedule = True
        args.decay_step = 25
        args.gamma = 1.0
        args.num_epochs = 5
        args.metric_ks = [5, 20]
        args.best_metric = 'Diversity@5'
 
        args.model_init_seed = 0

        args.bert_dropout = 0.20
        args.bert_hidden_units = 128 # embedding dimension
        args.bert_mask_prob = 0.30
        args.bert_max_len = 8
        args.bert_num_blocks = 2
        args.bert_num_heads = 4