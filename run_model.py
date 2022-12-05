import os
import torch
import torch.onnx
import numpy as np

from options import run_model_opts
from utils import logger, recorders, test_utils
from datasets import benchmark_loader
from models import build_model

args = run_model_opts.RunModelOpts().parse()
log  = logger.Logger(args)

def exprot_flow_model(model):
    x = [
        torch.randn(1, 3, 480, 640, requires_grad=True),
        torch.randn(1, 3, 480, 640, requires_grad=True),
        torch.randn(1, 3, 480, 640, requires_grad=True),
    ]
    torch.onnx.export(
        model.fnet.module.cpu(),
        x,
        'hdr_flownet.onnx',
        export_params=True,
        do_constant_folding=True,
        input_names=['frame_1', 'frame_2', 'frame_3'],
        output_names=['flow_1', 'flow_2'],
        dynamic_axes={
            'input' : {0 : 'batch_size'},    # variable length axes
            'output' : {0 : 'batch_size'}
        },
        opset_version=16,
    )

    if not os.path.exists("flownet_input_0.arr"):
        for i, xx in enumerate(x):
            xx = xx.cpu().detach().numpy()
            # xx = np.transpose(xx, [0, 2, 3, 1])
            binary = xx.tobytes()
            with open(f"flownet_input_{i}.arr", mode='wb') as f:
                f.write(binary)


def exprot_weight_model(model):
    hdr_cat_ldr = torch.randn(1, 30, 480, 640, requires_grad=True)
        
    merge_hdr = [
        torch.randn(1, 3, 480, 640, requires_grad=True),
        torch.randn(1, 3, 480, 640, requires_grad=True),
        torch.randn(1, 3, 480, 640, requires_grad=True),
        torch.randn(1, 3, 480, 640, requires_grad=True),
        torch.randn(1, 3, 480, 640, requires_grad=True),
    ]
    torch.onnx.export(
        model.mnet.module.cpu(),
        (hdr_cat_ldr, merge_hdr),
        'hdr_weight.onnx',
        export_params=True,
        do_constant_folding=True,
        input_names=['hdr_cat_ldr'] + [f"merge_hdr_{i}" for i in range(5)],
        output_names=['weights', 'hdr'],
        dynamic_axes={
            'input' : {0 : 'batch_size'},    # variable length axes
            'output' : {0 : 'batch_size'}
        },
        opset_version=16,
    )


def exprot_fine_model(model):
    inputs = {
        "x": torch.randn(1, 3, 6, 480, 640, requires_grad=True),
    }

    torch.onnx.export(
        model.mnet2.module.cpu(),
        (inputs['x'], inputs),
        'hdr_fine.onnx',
        export_params=True,
        do_constant_folding=True,
        input_names=['x'],
        output_names=['hdr'],
        dynamic_axes={
            'input' : {0 : 'batch_size'},    # variable length axes
            'output' : {0 : 'batch_size'}
        },
        opset_version=16,
    )


def dryrun_flownet(model):
    flownet = model.fnet.module.cpu()
    inputs = []
    for i in range(3):
        with open(f"flownet_input_{i}.arr", mode='rb') as f:
            tensor = np.frombuffer(f.read(), dtype=np.float32)
            tensor = tensor.reshape([1, 3, 480, 640])
            # tensor = np.transpose(tensor, (0, 3, 1, 2))
            tensor = torch.tensor(tensor)
            inputs.append(tensor)
    
    out = flownet(inputs)
    out = np.concatenate([out['flow1'], out['flow2']])
    # out = np.transpose(out, (0, 2, 3, 1))
    print(out)
    np.savez('flownet_pt_out.npz', pred=out)


def main(args):
    test_loader = benchmark_loader(args, log)
    model: torch.nn.DataParallel = build_model(args, log)
    recorder = recorders.Records()

    # test_utils.test(args, log, 'test', test_loader, model, 1, recorder)

    # log.plot_curves(recorder, 'test')
    model.eval()
    # exprot_flow_model(model)
    # exprot_weight_model(model)
    # exprot_fine_model(model)
    dryrun_flownet(model)

    

if __name__ == '__main__':
    from loguru import logger
    with logger.catch(reraise=True):
        torch.manual_seed(args.seed)
        main(args)
