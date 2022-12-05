# %%

import onnx
import numpy as np
import matplotlib.pyplot as plt
from onnx_tf.backend import prepare

# onnx_model_path = 'hdr_flownet.onnx'
# onnx_model = onnx.load(onnx_model_path)

# tf_model_path = 'tf_hdr_flownet'
# tf_rep = prepare(onnx_model)
# tf_rep.export_graph(tf_model_path)


# %%

with open('flownet_out', mode='rb') as f:
    out = np.frombuffer(f.read(), dtype=np.float32)
    out = out.reshape([2, 2, 480, 640])

pt_out = np.load('flownet_pt_out.npz')['pred']

# %%

plt.figure(figsize=(20, 20))
plt.imshow(out.reshape([4 * 480, 640]))

# %%
plt.figure(figsize=(20, 20))
plt.imshow(pt_out.reshape([4 * 480, 640]))
# %%