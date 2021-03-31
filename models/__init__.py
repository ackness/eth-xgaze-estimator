import importlib
import os

import timm

__all__ = ['get_model']


def get_model(config):
    all_model_list = [f[:-3] for f in os.listdir(os.path.dirname(__file__))]
    if 'timm_' in config.model_name:
        from timm_created_models import Model
        timm_all_pretrained_model_names = timm.list_models(pretrained=True)
        model_name = config.model_name[5:]
        assert model_name in timm_all_pretrained_model_names, "sry, this model is not in timm pretrained model list"
        model = Model(model_name, pretrained=config.pretrained)
    elif config.model_name in all_model_list:
        module = importlib.import_module(config.model_name)
        model = module.Model()
        # raise NotImplementedError("Please define your own model here!")
    else:
        raise NotImplementedError("Not find a proper model! Please check your configs file.")
    return model
