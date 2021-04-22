__all__ = ['get_dataloader']


def get_dataloader(config):
    if config.data_type in ['with-in', 'within_dataset', 'same']:
        from .xgaze import get_xgaze_train_loader, get_xgaze_test_loader, get_xgaze_train_val_loader
        if config.mode == 'train':
            if config.use_val:
                train_loader, val_loader = get_xgaze_train_val_loader(config)
                return train_loader, val_loader
            else:
                train_loader = get_xgaze_train_loader(config)
                return train_loader
        elif config.mode == 'test':
            test_loader = get_xgaze_test_loader(config)
            return test_loader
        else:
            raise ValueError("config.mode must in [train, test]")
    elif config.data_type in ['cross', 'cross_dataset', 'different']:
        raise NotImplementedError('We dont implement these part.')
    else:
        raise ValueError('Only support with-in or cross dataset type!')
