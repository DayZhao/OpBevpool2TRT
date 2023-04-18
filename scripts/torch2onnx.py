import torch
import torch.nn as nn
import argparse
from ops import bevpool
import onnx
import onnxruntime
import os 
import copy
import onnxsim
import numpy as np
from ops2trt.ops.bevpool.lss_view_transformer import LSSViewTransformer
from ops.utils.config import Config

ops = [bevpool]

class BaseModel(nn.Module):
    def __init__(self, op):
        super(BaseModel, self).__init__()
        
    def forward(self, *input):
        pass
    
class Torch2Onnx(nn.Module):
    def __init__(self, 
              model,
              torch_model_file,
              output_file,
              input_names,
              output_names,
              opset_version
              ):
        super(Torch2Onnx,self).__init__()
        if model is not None:
          self.model = model
        if torch_model_file is not None:
            self.model.load_state_dict(torch.load(torch_model_file), strict=True) 
            self.model.eval()
        self.output_file = output_file
        self.opset_version = opset_version
        self.input_names = input_names
        self.output_names = output_names

    def onnx_export(self, 
                    dummy_inputs=(torch.randn((1,2,120,120)))
                    ):
        #To do :Dynamic shape
        inpus_copy = copy.deepcopy(dummy_inputs)
        if not os.path.exists(self.output_file):
            torch.onnx.export(
                self.model,
                dummy_inputs,
                self.output_file,
                input_names=self.input_names,
                output_names=self.output_names,
                export_params=True,
                keep_initializers_as_inputs=True,
                do_constant_folding=True,
                opset_version=self.opset_version,
                dynamic_axes=None,
            )
            print(f"Successfully exported ONNX model: {self.output_file}")
        else:
            print(f"psss export  ONNX model: {self.output_file}")

        return None

def build_model(op):
    if op not in ops:
        raise ValueError("op not supported")

    if op == "bevpool":
        model = BaseModel(op)
        model.forward = bevpool.TRTTRTBEVPoolv2.apply
        
    return model

def get_dummy_input(op, **cfg):
    if op not in ops:
        raise ValueError("op not supported")
    
    if op == "bevpool":
        input_names = ["depth", "feat", "ranks_depth", "ranks_feat", "ranks_bev", "bev_feat_shape", "interval_starts", "interval_lengths"]
        output_names = ["bev_feat"]
        lss = LSSViewTransformer(cfg["grid_config"], cfg["input_size"], 1)
        b = cfg["bs"]
        c = cfg["channel_num"]
        h, w = cfg["input_size"]
        d = cfg["grid_config"]["intercal_num"]
        
        depth = np.random.rand(b, c, h, w).astype(np.float32)
        depth = np.exp(depth) / np.sum(np.exp(depth), axis=1, keepdims=True)
        feat = np.random.rand(b, h, w, d).astype(np.float32)
        
        inputs = lss.pre_op_input(cfg["bs"])
        
        inputs_npy = [i.cpu().numpy() for i in inputs]
        return (depth, feat, *inputs_npy), input_names, output_names        
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert torch model to onnx model')
    parser.add_argument("--config", default=None, help="The config file to convert")
    parser.add_argument("--fp16", action="store_true", help="Convert to fp16")
    parser.add_argument("--out_onnx_file", default=None, help="The output onnx file")
    args = parser.parse_args() 
    
    cfg = Config.fromfile(args.config)
    model = build_model(**cfg)
    dummy_inputs, input_names, output_names = get_dummy_input(cfg["op"], **cfg)
    
    Convertor = Torch2Onnx(model, None, args.out_onnx_file, input_names, output_names, 11)
    
    Convertor.onnx_export(dummy_inputs)
    
    
    









