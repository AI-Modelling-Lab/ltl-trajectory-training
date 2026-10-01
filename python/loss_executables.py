import torch
import subprocess
import re
import functools
import operator
import os
import array

def show_shape(data):
    return "[" + ", ".join(list(map(str, data.shape))) +"]"

def elements(t):
    if t.size() == ():
        return [t.item()]
    else:
        D = t.size(0)
        result = [e for i in range(0, D) for e in elements(t[i])]
        return result

def make_input(data):
    return array.array('d', elements(data)).tobytes()

def run_exp(executable, mode, gamma, constr, data):
    base = os.path.dirname(os.path.realpath(__file__)) # TODO better target executables
    full_executable = base + "/" + executable
    return subprocess.run([full_executable, mode, str(gamma), constr, show_shape(data)], input=make_input(data), capture_output=True)

def make_tensor(dims, vec):
    if len(dims) == 0:
        return torch.tensor([])
    elif len(dims) == 1:
        return torch.tensor(vec)
    else:
        d, *ds = dims
        def t(i):
            stride = functools.reduce(operator.mul, ds, 1)
            start = i * stride
            end = start + stride
            subvec = vec[start:end]
            return make_tensor(ds, subvec)
        ts = [t(i) for i in range(0,d)]
        return torch.stack(tuple(ts))

def parse_output(shape, output):
    data = array.array('d')
    data.frombytes(output)

    # Calculate expected numbers of elements
    lossN = functools.reduce(operator.mul, shape[2:], 1)
    gradN = functools.reduce(operator.mul, shape, 1)

    # Infer mode from number of output elements
    N = len(data)
    if N == lossN: # loss
        return data
    elif N == gradN: # grad
        return make_tensor(shape, data)
    elif N == lossN + gradN: # both
        loss = data[:lossN]
        grad = make_tensor(shape, data[lossN:])
        return (loss, grad)
    else:
        raise ValueError("Unexpected number of output elements: {} for shape {}".format(N, shape))
