import os
import sys

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

import argparse
import torch
from thop import profile
from thop.utils import clever_format
import time
import importlib
import copy

def parse_args():
    """
    args for training.
    """
    parser = argparse.ArgumentParser(description='Parse args for training')
    # for train
    parser.add_argument('--script', type=str, default='litetrack', choices=['litetrack'],
                        help='training script name')
    parser.add_argument('--config', type=str, default='B4_cae_center_got10k_ep100', help='yaml configure file name')
    args = parser.parse_args()

    return args


def evaluate_vit(model, template, search):
    '''Speed Test'''
    use_fp16 = False
    if use_fp16:
        template = template.half()
        search = search.half()
        # template_bb = template_bb.half()
    model_ = copy.deepcopy(model)
    device = torch.device(0)
    with torch.autocast(device.type, enabled=use_fp16):
        macs1, params1 = profile(model, inputs=(template, search),
                                custom_ops=None, verbose=False)
    macs, params = clever_format([macs1, params1], "%.3f")
    print('MACs: ', macs)
    print('Params: ', params)
    
    T_w = 200
    T_t = 500
    torch.cuda.synchronize()
    with torch.autocast(device.type, enabled=use_fp16):
        with torch.no_grad():
            # overall
            for i in range(T_w):
                _ = model_(template, search)
            start = time.time()
            for i in range(T_t):
                _ = model_(template, search)
            torch.cuda.synchronize()
            end = time.time()
            avg_lat = (end - start) / T_t
            print("Average latency: %.2f ms" % (avg_lat * 1000))
            print("FPS: %.2f" % (1. / avg_lat))




if __name__ == "__main__":
    device = "cuda:0"
    torch.cuda.set_device(device)
    # Compute the Flops and Params of our STARK-S model
    args = parse_args()
    '''update cfg'''
    yaml_fname = 'experiments/%s/%s.yaml' % (args.script, args.config)
    config_module = importlib.import_module('lib.config.%s.config' % args.script)
    cfg = config_module.cfg
    config_module.update_config_from_file(yaml_fname)
    '''set some values'''
    bs = 1
    z_sz = cfg.DATA.TEMPLATE.SIZE
    x_sz = cfg.DATA.SEARCH.SIZE

    if args.script == "ostrack":
        model_module = importlib.import_module('lib.models')
        model_constructor = model_module.build_ostrack
        model = model_constructor(cfg, training=False)
        # get the template and search
        template = torch.randn(bs, 3, z_sz, z_sz)
        template_bb = torch.tensor([[0.5,0.5,0.5,0.5]])
    
        # template = torch.randn(bs, 64, 768)
        search = torch.randn(bs, 3, x_sz, x_sz)
        # transfer to device
        model = model.to(device)
        model.eval()
        template = template.to(device)
        search = search.to(device)
        template_bb = template_bb.to(device)
        evaluate_vit(model, template, search, template_bb)
 

    if args.script == "litetrack":
        model_module = importlib.import_module('lib.models')
        model_constructor = model_module.build_LiteTrack
        model = model_constructor(cfg, training=False)

        # get the template and search
        
        # template = torch.randn(bs, 3, z_sz, z_sz)
        # template_bb = torch.tensor([[0.5,0.5,0.5,0.5]])
        template = torch.randn(bs, 64, 768)
        search = torch.randn(bs, 3, x_sz, x_sz)

        # transfer to device
        model = model.to(device)
        model.eval()
        template = template.to(device)
        search = search.to(device)

        evaluate_vit(model, template, search)



    else:
        raise NotImplementedError
