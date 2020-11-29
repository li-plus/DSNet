from anchor_based.dsnet import DSNet
from anchor_free.dsnet_af import DSNetAF


def get_anchor_based(base_model, num_feature, num_hidden, anchor_scales,
                     num_head, **kwargs):
    return DSNet(base_model, num_feature, num_hidden, anchor_scales, num_head)


def get_anchor_free(base_model, num_feature, num_hidden, num_head, **kwargs):
    return DSNetAF(base_model, num_feature, num_hidden, num_head)


def get_model(model_type, **kwargs):
    if model_type == 'anchor-based':
        return get_anchor_based(**kwargs)
    elif model_type == 'anchor-free':
        return get_anchor_free(**kwargs)
    else:
        raise ValueError(f'Invalid model type {model_type}')
