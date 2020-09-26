import torch
import math


class Identity(torch.nn.Module):
    def forward(self, input):
        return input


shape_global = 0


class SegmentConsensus(torch.autograd.Function):
    def __init__(self, dim=1):
        global shape_global
        shape_global = None

    @staticmethod
    def forward(ctx, input_tensor):
        global shape_global
        shape_global = input_tensor.size()
        output = input_tensor.mean(dim=1, keepdim=True)
        ctx.save_for_backward(output)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_in = grad_output.expand(shape_global) / float(shape_global[1])

        return grad_in


class ConsensusModule(torch.nn.Module):
    def __init__(self, dim=1):
        super(ConsensusModule, self).__init__()

    def forward(self, input):
        return SegmentConsensus().apply(input)
