import onnx
from onnx_tf.backend import prepare

onnx_model_path = 'hdr_flownet.onnx'
onnx_model = onnx.load(onnx_model_path)

tf_model_path = 'tf_hdr_flownet'
tf_rep = prepare(onnx_model)
tf_rep.export_graph(tf_model_path)