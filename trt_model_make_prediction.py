# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 13:01:26 2022

@author: faithwh14
"""

import pycuda 
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt 
import numpy as np

saved_trt_path = "liho_engine.trt"
f = open(saved_trt_path, "rb")
TRT_LOGGER = trt.Logger()
runtime = trt.Runtime(TRT_LOGGER)
engine = runtime.deserialize_cuda_engine(f.read())
context = engine.create_execution_context()

h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype = trt.nptype(trt.float32))
h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype = trt.nptype(trt.float32))
d_input = cuda.mem_alloc(h_input.nbytes)
d_output = cuda.mem_alloc(h_output.nbytes)
stream = cuda.Stream()

xs = np.load("testing_set.npy")
x = [xs[i].astype(trt.nptype(trt.float32)).ravel() for i in range(len(xs))] #read_img().astype(trt.nptype(trt.float32)).ravel()

y_pred = np.zeros([len(x), 2])

for h, i in enumerate(x):
    np.copyto(h_input, i)
    cuda.memcpy_htod_async(d_input, h_input, stream)
    context.execute_async(bindings = [int(d_input), int(d_output)], stream_handle = stream.handle)
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    stream.synchronize()

    y_pred[h, 0] = h_output[0]
    y_pred[h, 1] = h_output[1]

np.save("y_prediction", y_pred)




	
