import time

import torch
from torch.autograd import Function
import torch.autograd as autograd

# TODO temporary hack to import the loss executable library
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'python'))
from loss_executables import run_exp, parse_output

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

# Returns PyTorch-compatible loss function for LTL constraint using generated executables
#
# Arguments:
#     ltl - LTL Constraint
#     consts - constant values to append after state vector
#     gamma - smoothing factor for approximation of non-differentiable functions
#     exectype - executable to use (haskell, ocaml or ocaml_torch)
def get_external_loss_fn(ltl, consts=[], gamma=0.005, exectype="ocaml"):

    def run_executable(mode, data):
        result = run_exp("ltl-loss-" + exectype, mode, gamma, ltl, data)
        if result.stderr:
            raise RuntimeError('Loss executable produced errors: {}'.format(result.stderr))
        return parse_output(list(data.shape), result.stdout)

    class LTL_Loss(Function):

        @staticmethod
        def forward(ctx, x):
            # Append constants to state vectors
            constsT = (torch.Tensor(consts)).repeat(list(x.shape[:2]) + [1]).to(DEVICE)
            y = torch.cat([x, constsT], dim = -1).to(DEVICE)

            # Permute dimensions to expected layout: time x state x batch
            z = y.permute(1, 2, 0)

            # Run executable to compute loss and gradient
            (loss, gradT) = run_executable("both", z)

            # Undo permutation in gradient and strip appended constants
            stateN = x.shape[2]
            gradT = gradT.permute(2, 0, 1)[:,:,:stateN]

            # Save gradients to be used in backward
            ctx.save_for_backward(gradT)

            # Return loss as a 0D tensor
            return torch.Tensor(loss).squeeze(0)

        @staticmethod
        def backward(ctx, grad_output):
            # Use the saved gradients
            x = ctx.saved_tensors[0]
            grad_input = autograd.Variable(x, requires_grad=True).to(DEVICE)
            return grad_input * grad_output

    return LTL_Loss

class Box:
    """ A domain defined by a bounding box with min point a and max point b"""
    def __init__(self, a, b):
        self.a = a
        self.b = b
        assert not self.is_empty()

    def is_empty(self):
        return torch.any(self.a > self.b)

    def project(self, x):
        return torch.min(torch.max(x, self.a), self.b)

    def sample(self):
        s = self.a + (self.b - self.a) * torch.rand_like(self.a)
        return s

def fully_global_ins(ins, epsilon):
    low_ins = ins - epsilon
    high_ins = ins + epsilon
    return [Box(low_ins[:, i], high_ins[:, i]) for i in range(ins.shape[1])]

# LTL constraint with optional:
# - transform to apply to trajectory,
# - constants appended to state vector in transformed trajectory,
# - smoothing factor,
# - executable to be used
class Constraint:
    def __init__(self, ltl, transform=None, consts=[], gamma=0.05, exectype="ocaml"):
        self.ltl = ltl
        self.transform = (lambda x : x) if transform is None else transform
        self.consts = consts
        self.gamma = gamma
        self.exectype=exectype

    # Loss function, expecting the trajectory as a tensor batch x time x state
    def loss(self, traj):
        loss_f = get_external_loss_fn(self.ltl, self.consts, self.gamma, self.exectype)

        # Transform trajectory
        x = self.transform(traj)

        # Time application of loss function
        #t0 = time.time()
        loss_applied = loss_f.apply(x)
        #print("Loss time: ", time.time() - t0)

        return loss_applied

    # TODO Doc
    def domains(self, ins, targets):
        return fully_global_ins(ins, 0.01) # TODO extract magic number

