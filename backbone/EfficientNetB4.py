import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone.geffnet.gen_efficientnet import tf_efficientnet_b4_ns


def EfficientNetB4(pretrained=False, num_classes=1000, act_layer=nn.ReLU, cfg_path = ""):    
    return loadstate(tf_efficientnet_b4_ns(pretrained = False, num_classes = num_classes, drop_rate=0.2, act_layer=act_layer), pretrained, cfg_path)
 
def Freq_B4(model):
    blocks = nn.ModuleList()
    blocks.append(model.blocks[0])
    blocks.append(model.blocks[1])
    path = "Pre-training weight path for EfficientNetB4"
    state_dict = torch.load(path, map_location='cpu')
    strict = True
    blocks.load_state_dict(state_dict, strict=strict)
    return blocks

def switch_layers(model, layers, x, tip=0, endtip=0):
    if 'stem' in layers:
        x = model.conv_stem(x)
        x = model.bn1(x)
        x = model.act1(x)    
        return x
    
    if 'blocks' in layers:
        for i in range(tip, endtip+1):
            x = model.blocks[i](x)
        return x
 
    if 'head' in layers:
        x = model.conv_head(x)
        x = model.bn2(x)
        x = model.act2(x)
        return x

    if 'fc' in layers:
        x = model.global_pool(x)
        x = x.flatten(1)
        if model.drop_rate > 0.:
            x = F.dropout(x, p=model.drop_rate, training=model.training)
        return x, model.classifier(x)
    
    if 'patch' in layers:
        x = model.conv_stem(x)   
        return x
    
    if 'norm' in layers:
        x = model.bn1(x)
        x = model.act1(x) 
        return x
    
        
    
def loadstate(model, pretrained, cfg_path):
    if pretrained:
        state_dict = torch.load(cfg_path, map_location='cpu')
        classifier = 'classifier'
        num_classes = getattr(model, classifier).weight.shape[0]
        classifier_weight = classifier + '.weight'
        pretrained_num_classes = state_dict[classifier_weight].shape[0]
        strict=True
        if num_classes != pretrained_num_classes:
            print('=> Discarding pretrained classifier since num_classes != {}'.format(pretrained_num_classes))
            del state_dict[classifier_weight]
            del state_dict[classifier + '.bias']
            strict = False
        model.load_state_dict(state_dict, strict=strict)
    return model

if __name__ == '__main__':
    model = EfficientNetB4(pretrained=True ,num_classes=1)
    x = torch.rand((4,3,256,256))
    # x = switch_layers(model, 'stem', x)
    # x = switch_layers(model, 'blocks', x, 0, 6)
    # x = switch_layers(model, 'head', x)
    # fea, x = switch_layers(model, 'fc', x)
    # # print()
    # block = freq_blocks(model, False)
    # print(block)
"""
    layer   c       h   w

    Stem    48      H/2 W/2
    block0  24      H/2 W/2
    block1  32      H/4 W/4
    block2  56      H/8 W/8
    block3  112     H/16 W/16
    block4  160     H/16 W/16
    block5  272     H/32 W/32
    block6  448     H/32 W/32
    head    1792    H/32 W/32
"""