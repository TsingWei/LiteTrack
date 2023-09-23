import os
import sys

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

import argparse
import torch
from lib.utils.misc import NestedTensor
from thop import profile
from thop.utils import clever_format
import time
import importlib
from thop.vision.basic_hooks import count_linear
import thop
from torch.nn import Linear
import copy
from torch.profiler import profile as t_profile, record_function, ProfilerActivity
# from onnxmltools.utils import float16_converter
import numpy as np
import onnxruntime as ort
from onnxconverter_common import float16
import onnx
# Register the hook for Linear layers


def parse_args():
    """
    args for training.
    """
    parser = argparse.ArgumentParser(description='Parse args for training')
    # for train
    parser.add_argument('--script', type=str, default='ostrack', choices=['ostrack'],
                        help='training script name')
    parser.add_argument('--config', type=str, default='vitb_384_mae_32x4_ep300', help='yaml configure file name')
    args = parser.parse_args()

    return args


def evaluate_vit(model, template, search, template_bb):
    '''Speed Test'''
    model_ = copy.deepcopy(model)
    
    macs1, params1 = profile(model, inputs=(template, search),
                             custom_ops=None, verbose=False)
    macs, params = clever_format([macs1, params1], "%.3f")
    print('overall macs is ', macs)
    print('overall params is ', params)

    output_onnx_name = 'test_net.onnx'

    inputs=(template, search,)
    torch.onnx.export(model, 
        inputs,
        output_onnx_name, 
        input_names=[ "zf", "x"], 
        output_names=["output"],
        opset_version=11,
        export_params=True,
        # verbose=True,
        # dynamic_axes={'input':{0:'batch', 2:'h', 3:'w'}, 'output':{0:'batch', 2:'h2', 3:'w2'}} 
    )

    providers = ['CUDAExecutionProvider']
    model = onnx.load("test_net.onnx")
    dtype = np.float16
    model_fp16 = float16.convert_float_to_float16(model)
    onnx.save(model_fp16, "test_net_fp16.onnx")
    ort_session = ort.InferenceSession("test_net_fp16.onnx", providers=providers)
    output = ort_session.run(output_names=['output'],
                             	input_feed={'zf':  np.array(template.cpu(), dtype=dtype),'x': np.array(search.cpu(), dtype=dtype)}
                                )

    
    T_w = 50
    T_t = 100
    print("testing speed ...")
    # torch.cuda.synchronize()
    # with torch.no_grad():
    #     # overall
    for i in range(T_w):
        output = ort_session.run(output_names=['output'],
                             	input_feed={'zf':  np.array(template.cpu(), dtype=dtype),'x': np.array(search.cpu(), dtype=dtype)}
                                )
    start = time.time()
    for i in range(T_t):
        output = ort_session.run(output_names=['output'],
                             	input_feed={'zf':  np.array(template.cpu(), dtype=dtype),'x': np.array(search.cpu(), dtype=dtype)}
                                )
    torch.cuda.synchronize()
    end = time.time()
    avg_lat = (end - start) / T_t
    print("The average overall latency is %.2f ms" % (avg_lat * 1000))
    print("FPS is %.2f fps" % (1. / avg_lat))
        # for i in range(T_w):
        #     _ = model(template, search)
        # start = time.time()
        # for i in range(T_t):
        #     _ = model(template, search)
        # end = time.time()
        # avg_lat = (end - start) / T_t
        # print("The average backbone latency is %.2f ms" % (avg_lat * 1000))
    # with t_profile(activities=[
    #     ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    #     with record_function("model_inference"):
    #         model_(template, search)
    # print(prof.key_averages(group_by_stack_n=5).table(sort_by="cpu_time_total", row_limit=100))



def get_data(bs, sz):
    img_patch = torch.randn(bs, 3, sz, sz)
    att_mask = torch.rand(bs, sz, sz) > 0.5
    return NestedTensor(img_patch, att_mask)


if __name__ == "__main__":
    device = "cpu"
    # torch.cuda.set_device(device)
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

        merge_layer = cfg.MODEL.BACKBONE.MERGE_LAYER
        if merge_layer <= 0:
            evaluate_vit(model, template, search, template_bb)
        else:
            evaluate_vit_separate(model, template, search)

    else:
        raise NotImplementedError
