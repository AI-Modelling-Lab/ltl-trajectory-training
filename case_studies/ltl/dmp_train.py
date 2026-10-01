import torch
from torch import nn, optim, autograd
import os
from os.path import join
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from datetime import datetime
import time

from .dmp_nn import DMPNN
from .dmp import *

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

total_time = [0.0, 0.0]

def evaluate_constraint(ins, targets, constraint, net, rollout_func):
    weights = net(ins)
    rollout_traj = rollout_func(ins[:, 0], ins[:, -1], weights)[0]
    return constraint.loss(rollout_traj)

def np_to_pgpu(np_array):
    t = torch.from_numpy(np_array).to(dtype=torch.float)
    if USE_CUDA:
        return t.cuda()

    return t

def plot_train_val(train_path, val_path):
    train_losses = np.loadtxt(train_path)
    val_losses = np.loadtxt(val_path)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    loss_metrics = train_losses.shape[1]
    for i in range(loss_metrics):
        plt.subplot(1, loss_metrics, i + 1)
        plt.plot(train_losses[:, i], label="train")
        plt.plot(val_losses[:, i], label="test")
        plt.legend()
    plt.show()

def t_stamp():
    return datetime.now().strftime("%Y-%m-%d")

def rollout_error(output_roll, target_roll):
    return torch.norm(output_roll - target_roll, dim=2).mean()

def load_dmp_demos(demos_folder):
    start_state_paths = sorted([d for d in os.listdir(demos_folder) if "start-state" in d])
    rollout_paths = sorted([d for d in os.listdir(demos_folder) if "rollout" in d])

    start_states = np.stack([np.loadtxt(join(demos_folder, sp), ndmin=2) for sp in start_state_paths])
    rollouts = np.stack([np.loadtxt(join(demos_folder, rp), ndmin=2) for rp in rollout_paths])

    # Note: May need to make this configurable later
    interp_rollout = interpolated_path_batch(rollouts, 100)

    return start_states, interp_rollout

def interpolated_path(recorded_ys, T):
    num_points, dims = recorded_ys.shape[0], recorded_ys.shape[1]
    path = np.zeros((dims, T))
    x = np.linspace(0, 1, num_points)

    dt = 1.0 / (T - 1)

    for d in range(dims):
        path_gen = interp1d(x, recorded_ys[:, d])
        path[d] = path_gen([t*dt for t in range(T)])

    return path

def interpolated_path_batch(recorded_ys, T):
    batch_size, num_points, dims = recorded_ys.shape
    path = np.zeros((batch_size, T, dims))
    x = np.linspace(0, 1, num_points)

    dt = 1.0 / (T - 1)

    for b in range(batch_size):
        for d in range(dims):
            path_gen = interp1d(x, recorded_ys[b, :, d])
            path[b, :, d] = path_gen([t*dt for t in range(T)])

    return path

def imitate_path(y_d, dmp):
    # Set initial state and goal
    y_start = y_d[0, :].copy()
    y_goal = y_d[-1, :].copy()

    x_track = canonical_rollout(dmp.cs.start_x, dmp.cs.ax, dmp.dt)

    # generate function to interpolate the desired trajectory
    path = interpolated_path(y_d, dmp.T)
    dims =  path.shape[0]

    # Calculate the velocity of y_des
    dy_d = np.diff(path) / dmp.dt

    # Add zero to the beginning of every row
    dy_d = np.hstack((np.zeros((dims, 1)), dy_d))

    # calculate the acceleration of y_des
    ddy_d = np.diff(dy_d) / dmp.dt
    ddy_d = np.hstack((np.zeros((dims, 1)), ddy_d))

    K = dmp.ay * dmp.by
    D = dmp.ay

    # Original form...
    # f_target = (ddy_d.T - (K * (y_goal - path.T) - D * dy_d.T))  / (dmp.goal - dmp.y0)

    # Modified general one
    f_target = (ddy_d.T + D * dy_d.T) / K - (dmp.goal - path.T) + (dmp.goal - dmp.y0) * np.tile(x_track, (dims, 1)).T

    # efficiently generate weights to realize f_target
    weights = gen_weights(f_target, y_start, y_goal, dmp)

    return path, weights


def gen_weights(f_target, y_start, y_goal, dmp):

    # calculate x and psi
    x_track = canonical_rollout(dmp.cs.start_x, dmp.cs.ax, dmp.dt)
    psi_track = rbf(x_track, dmp.h, dmp.c)

    # efficiently calculate BF weights using weighted linear regression
    weights = np.zeros((dmp.dims, dmp.n_basis_funcs))
    for d in range(dmp.dims):
        # spatial scaling term
        # k = (y_goal[d] - y_start[d])
        for b in range(dmp.n_basis_funcs):
            weights[d,b] = (np.sum(x_track * psi_track[:, b] * f_target[:, d])) / (np.sum(x_track**2*psi_track[:, b]))
    weights = np.nan_to_num(weights)
    return weights


def save_dmp(dmp, dmp_path):
    with open(dmp_path, "wb") as f:
        pickle.dump(dmp, f, 2)


def load_dmp(dmp_path):
    with open(dmp_path,"rb") as f:
        dmp = pickle.load(f)
    return dmp


def plot_rollout(dmp_rollout, raw_path = None):
    dims = dmp_rollout.shape[1]
    for d in range(dims):
        plt.subplot(2,ceil(dims / 2),d + 1)
        dmp_timescale = np.linspace(0, 1, dmp_rollout.shape[0])
        plt.plot(dmp_timescale, dmp_rollout[:, d], label="DMP")

        if raw_path is not None:
            raw_timescale = np.linspace(0, 1, raw_path.shape[0])
            plt.plot(raw_timescale, raw_path[:, d], label="Raw")
        plt.legend()
    plt.show()

def display_timediff(msg, diff):
    print("{} : {} hr {} min {} sec".format(msg,int(diff//3600),int((diff%3600)//60), int(diff%60)))

# Train a model with corresponding DMP on training (and optionally validation) data, with an optional constraint
# - nu: bias for loss combination, with 1.0 being equal to imitation loss (default: 1.0)
# - N: number of iterations to run (default: 200)
# - report_step: iteration period for time reporting (default: 10)
def training_loop(model, dmp, train_set, val_set, constraint, adversarial, nu=1.0, N=200, report_step=10):
    optimizer = optim.Adam(model.parameters())
    loss_fn = rollout_error
    train_loader = DataLoader(train_set, shuffle=True, batch_size=32)
    if val_set is not None:
        val_loader = DataLoader(val_set, shuffle=False, batch_size=32)
    else:
        val_loader = None

    train_losses, val_losses = [], []

    def batch_learn(data_loader, adv, optimize=False):
        losses = []
        for batch_idx, (starts, rollouts) in enumerate(data_loader):
            batch_size, T, dims = rollouts.shape

            learned_weights = model(starts)
            learned_rollouts = dmp.rollout_torch(starts[:, 0], starts[:, -1], learned_weights)[0]

            main_loss = loss_fn(learned_rollouts, rollouts).to(DEVICE)

            if constraint is None:
                c_loss = torch.tensor(0.0).to(DEVICE)
            else:
                c_loss = evaluate_constraint(starts, rollouts, constraint, model, dmp.rollout_torch)

            full_loss = 1.0 * main_loss + nu * c_loss

            losses.append([main_loss.item(), c_loss.item(), full_loss.item()])

            if optimize:
                optimizer.zero_grad()
                full_loss.backward()
                optimizer.step()

        return np.mean(losses, 0, keepdims=True)

    last_t = time.time()

    for epoch in range(N):

        # Time execution
        if epoch % report_step == 0 and epoch / report_step > 0:
            dd = time.time()-last_t
            remaining_N = N-epoch
            remaining_t = remaining_N / float(report_step) * dd
            print("Iteration {}/{}".format(epoch, N))
            display_timediff("Time for last set of {}".format(report_step), dd)
            display_timediff("Approx. time to completion",remaining_t)
            last_t = time.time()

        # Train loop
        model.train()
        avg_train_loss = batch_learn(train_loader, adversarial, True)
        train_losses.append(avg_train_loss[0])

        # Validation Loop
        if val_loader is not None:
            model.eval()
            avg_val_loss = batch_learn(val_loader, False, False)
            val_losses.append(avg_val_loss[0])

    return train_losses, val_losses

# Plot DMPNN model rolled out using DMP against a sample of parameters and rollout
def plot_vs_sample(model, dmp, sample, size=(8, 8), title=None, ann_f=None):
    with plt.style.context('seaborn-v0_8'):
        # Get the plot and axes
        fig, ax = plt.subplots()

        # Split first training sample and scatter its rollout
        train_starts, train_rollout = sample
        scatter_rollout(ax, train_rollout.detach().cpu(), "Sample", 'orange')

        # Switch model to evaluation, get learned DMP weights for the same inputs, compute DMP rollout and scatter it
        model.eval()
        learned_weights = model(train_starts.unsqueeze(0))
        learned_dmp_rollout = dmp.rollout_torch(train_starts[0].unsqueeze(0), train_starts[-1].unsqueeze(0), learned_weights, 1.0)[0][0].detach().cpu()
        scatter_rollout(ax, learned_dmp_rollout, "Learned")

        # Format the plot and add any extra annotation
        format_plot(fig, ax, size = size, title = title)
        if ann_f is not None:
            ann_f(plt, fig, ax)

        # Show the current plot
        plt.show()

# Plot two DMPNN models rolled out using the same DMP against each other
def plot_models(model1, model2, dmp, sample, size=(8, 8), title=None, ann_f=None):
    with plt.style.context('seaborn-v0_8'):
        # Get the plot and axes
        fig, ax = plt.subplots()

        # Split first training sample and scatter its rollout
        train_starts, train_rollout = sample

        # For each model: switch to evaluation, get learned DMP weights for the same inputs, compute DMP rollout and scatter it
        model1.eval()
        learned_weights1 = model1(train_starts.unsqueeze(0))
        learned_dmp_rollout1 = dmp.rollout_torch(train_starts[0].unsqueeze(0), train_starts[-1].unsqueeze(0), learned_weights1, 1.0)[0][0].detach().cpu()
        scatter_rollout(ax, learned_dmp_rollout1, "Model 1")

        model2.eval()
        learned_weights2 = model2(train_starts.unsqueeze(0))
        learned_dmp_rollout2 = dmp.rollout_torch(train_starts[0].unsqueeze(0), train_starts[-1].unsqueeze(0), learned_weights2, 1.0)[0][0].detach().cpu()
        scatter_rollout(ax, learned_dmp_rollout2, "Model 2")

        # Format the plot and add any extra annotation
        format_plot(fig, ax, size = size, title = title)
        if ann_f is not None:
            ann_f(plt, fig, ax)

        # Show the current plot
        plt.show()

# Format the plot for model visualisation
# - fig, ax: the figure and axes to format
# - size: plot size in inches (default: (8, 8))
# - title: title for the plot (default: None)
def format_plot(fig, ax, size=(8, 8), title=None):
    fig.set_size_inches(size)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    fig.legend(prop={"size": 16})
    fig.tight_layout()
    ax.tick_params(labelsize=14)

    if title is not None:
        ax.set_title(title, fontsize=16)

# Scatter a rollout (from training data or DMP with learned weights) into a plot
def scatter_rollout(ax, rollout, label, colour=None, alpha=0.5):
    ax.scatter(rollout[:, 0], rollout[:, -1], label=label, c=colour, alpha=alpha)

# Plot reported training and validation losses
def plot_losses(train_losses, val_losses):
    with plt.style.context('seaborn-v0_8'):
        # Get the plot and axes
        fig, ax = plt.subplots()

        # Merge and plot training losses
        tl = np.stack(train_losses, axis=0)
        ax.plot(tl[:,0], label="Imitation Loss (Training)")
        ax.plot(tl[:,1], label="Constraint Loss (Training)")
        ax.plot(tl[:,2], label="Total Loss (Training)")

        # Merge and plot validation losses
        vl = np.stack(val_losses, axis=0)
        ax.plot(vl[:,0], label="Imitation Loss (Validation)")
        ax.plot(vl[:,1], label="Constraint Loss (Validation)")
        ax.plot(vl[:,2], label="Total Loss (Validation)")

        # Add legend and show the plot
        fig.legend()
        plt.show()

# Load training and validation datasets
# - demo_folder: folder with training and validation data
# - i: data point to select (default: 0 - first)
#
# TODO: note that this selects a single training sample!
def load_data(demo_folder, i=0):
    t_start_states, t_pose_hists = load_dmp_demos(demo_folder + "/train")
    t_start_states, t_pose_hists = np_to_pgpu(t_start_states), np_to_pgpu(t_pose_hists)
    train_set = TensorDataset(t_start_states[i:i+1], t_pose_hists[i:i+1])
    print("Train set length: {}".format(len(train_set)))

    v_start_states, v_pose_hists = load_dmp_demos(demo_folder + "/val")
    v_start_states, v_pose_hists = np_to_pgpu(v_start_states), np_to_pgpu(v_pose_hists)
    val_set = TensorDataset(v_start_states[i:i+1], v_pose_hists[i:i+1])

    return train_set, val_set

# Create DMPNN model and accompanying DMP
# - in_dim: ??? (DMPNN parameter)
# - dmp_dim: ??? (DMPNN and DMP parameter)
# - basis_fs: ??? (DMPNN and DMP parameter) (default: 30)
# - dt: ??? (DMP parameter) (default: 0.01)
def create_model(in_dim, dmp_dim, basis_fs=30, dt=0.01):
    dmp = DMP(basis_fs, dt, dmp_dim, DEVICE)
    if USE_CUDA:
        model = DMPNN(in_dim, 1024, dmp_dim, basis_fs).cuda()
    else:
        model = DMPNN(in_dim, 1024, dmp_dim, basis_fs)

    return model, dmp

# create_model but with in_dim and dmp_dims determined by a training set (resp. its first sample)
def create_model_for_train_set(train_set, basis_fs=30, dt=0.01):
    in_dim = train_set[0][0].numel()
    dmp_dim = train_set[0][1].shape[1]
    return create_model(in_dim, dmp_dim, basis_fs, dt)

def train(demo_type, demo_folder, constraint=None, results_root=None, ann_f=None, out_img=None, nu=1.0):
    # Print header: demo type and constraint when given
    print("{}, {}".format(demo_type, constraint.ltl if constraint is not None else "unconstrained"))

    # Load data, build model, train it, plot against a training sample
    train_set, val_set = load_data(demo_folder)
    model, dmp = create_model_for_train_set(train_set)
    train_losses, val_losses = training_loop(model, dmp, train_set, val_set, constraint, False, nu)
    plot_vs_sample(model, dmp, train_set[0], title=demo_type, ann_f=ann_f)
    # TODO save the plot if out_img is not None

    # Save final model and losses if requested
    if results_root is not None:
        # Make sure results folder exists if requested
        results_folder = join(results_root, "{}".format(demo_type))
        os.makedirs(results_folder, exist_ok=True)

        torch.save(model.state_dict(), join(results_folder, "learned_model_epoch_final.pt"))
        np.savetxt(join(results_folder, "train_losses.txt"), train_losses)
        np.savetxt(join(results_folder, "val_losses.txt"), val_losses)
