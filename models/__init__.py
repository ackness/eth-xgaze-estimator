import importlib
import os

import timm

__all__ = ['get_model']


def get_model(config):
    all_model_list = [f[:-3] for f in os.listdir(os.path.dirname(__file__))]

    if 'timm_' in config.model_name:
        print(f"Using pretrained timm model {config.model_name}")
        from .timm_created_models import Model
        timm_all_pretrained_model_names = timm.list_models(pretrained=True)
        model_name = config.model_name[5:]
        assert model_name in timm_all_pretrained_model_names, "sry, this model is not in timm pretrained model list"
        model = Model(model_name, pretrained=config.timm_pretrained)

    elif config.model_name in all_model_list:
        print(f"Using model {config.model_name}")
        module = importlib.import_module(f".{config.model_name}", package="models")
        model = module.Model(config)
        # raise NotImplementedError("Please define your own model here!")

    else:
        raise NotImplementedError(f"Not find a proper model={config.model_name}! Please check your configs file.")
    return model


if __name__ == '__main__':
    all_model_list = [f[:-3] for f in os.listdir(os.path.dirname(__file__))]
    print(all_model_list)
