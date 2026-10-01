# The case studies from our notebooks, but meant to be run outside of Jupyter (e.g. for running a batch on a server)

# --- Imports, constants and helper functions
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import ltl.dmp_train as LTL # DMP training from Innes and Ramamoorthy with our generated loss function
from ltl.constraint import Constraint
from datetime import datetime
import time
from pathlib import Path
import random
import os
import sys
from plane_navigation import *
from pathlib import Path

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
LOG = Path.cwd() / 'case_studies_{}.log'.format(datetime.now())
FIGS = Path.cwd() / 'figures_{}'.format(datetime.now())

def log(message, end='\n'):
    print(message, end=end)
    with LOG.open(mode='a') as f:
        f.write(message + end)

def t_stamp():
    return datetime.now().strftime("%Y-%m-%d-%H")

def runtime(t0):
    diff = time.time()-t0
    log("Runtime : {} hr {} min {} sec".format(int(diff//3600),int((diff%3600)//60), int(diff%60)))

def L2(components_tensor):
    """
    Compute the L2 (Euclidean) norm of a tensor whose final axis
    is the state component dimension. E.g. a trajectory (n,d) with n sample
    points and d coordinates per point.
    """
    return torch.sqrt(torch.sum(components_tensor**2, dim=-1))

# --- Plotting

# Format the plot for model visualisation
# - fig, ax: the figure and axes to format
# - size: plot size in inches (default: (8, 8))
# - title: title for the plot (default: None)
def format_plot(fig, ax, size=(8, 8), title=None, legend=True):
    fig.set_size_inches(size)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel('x')
    plt.xticks(np.arange(0.0, 1.1, 0.1))
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel('y')
    plt.yticks(np.arange(0.0, 1.1, 0.1))
    if legend:
        ax.legend(prop={"size": 16}, loc='upper left')
    fig.tight_layout()
    ax.tick_params(labelsize=14)
    ax.grid()

    if title is not None:
        ax.set_title(title, fontsize=16)

# Plot a trajectory into a plot (from data, DMP, trained model, ...)
def plot_trajectory(ax, points, symbol, label, colour=None, alpha=0.5, linewidth=1):
    ax.plot(points[:, 0], points[:, -1], symbol, label=label, c=colour, alpha=alpha)

def plot_one_trajectory(t0, n0, size=(8,8), title=None, ann_f=None, legend=True):
    with plt.style.context('default'):
        # Get the plot and axes
        fig, ax = plt.subplots()

        # Plot both trajectories
        plot_trajectory(ax, t0, 'o-', n0, colour='orange') # TODO make colour parameter

        # Standard formatting of the plot
        format_plot(fig, ax, size = size, title = title, legend = legend)

        # Any further annotation of the plot
        if ann_f is not None:
            ann_f(plt, fig, ax)

        return (fig, ax)

def plot_two_trajectories(t0, n0, t1, n1, size=(8,8), title=None, ann_f=None, legend=True):
    with plt.style.context('default'):
        # Get the plot and axes
        fig, ax = plt.subplots()

        # Plot both trajectories
        plot_trajectory(ax, t0, 'o-', n0, colour='orange') # TODO make colour parameter
        plot_trajectory(ax, t1, '^-', n1, colour=None) # TODO make colour parameter

        # Standard formatting of the plot
        format_plot(fig, ax, size = size, title = title, legend = legend)

        # Any further annotation of the plot
        if ann_f is not None:
            ann_f(plt, fig, ax)

        return (fig, ax)

# Plot DMPNN model rolled out using DMP against a sample of parameters and rollout
def plot_vs_sample(model, dmp, sample, size=(8, 8), title=None, ann_f=None, legend=True):
    # Split first training sample
    train_starts, train_rollout = sample

    # Switch model to evaluation, get learned DMP weights for the same inputs, compute DMP rollout
    model.eval()
    learned_weights = model(train_starts.unsqueeze(0))
    learned_dmp_rollout = dmp.rollout_torch(train_starts[0].unsqueeze(0), train_starts[-1].unsqueeze(0), learned_weights, 1.0)[0][0].detach().cpu()

    # Plot both training and learned rollouts
    (fig, ax) = plot_two_trajectories(train_rollout.detach().cpu(), "Training", learned_dmp_rollout, "Learned", size=size, title=title, ann_f=ann_f, legend=legend) # TODO make trajectory titles a parameter

    return fig

# Plot two DMPNN models rolled out using the same DMP against each other
def plot_models(model1, title1, model2, title2, dmp, sample, size=(8, 8), title=None, ann_f=None, legend=True):
    # Split first training sample
    train_starts, train_rollout = sample

    # For each model: switch to evaluation, get learned DMP weights for the same inputs, compute DMP rollout
    model1.eval()
    learned_weights1 = model1(train_starts.unsqueeze(0))
    learned_dmp_rollout1 = dmp.rollout_torch(train_starts[0].unsqueeze(0), train_starts[-1].unsqueeze(0), learned_weights1, 1.0)[0][0].detach().cpu()

    model2.eval()
    learned_weights2 = model2(train_starts.unsqueeze(0))
    learned_dmp_rollout2 = dmp.rollout_torch(train_starts[0].unsqueeze(0), train_starts[-1].unsqueeze(0), learned_weights2, 1.0)[0][0].detach().cpu()

    # Plot both learned rollouts
    (fig, ax) = plot_two_trajectories(learned_dmp_rollout1, title1, learned_dmp_rollout2, title2, size=size, title=title, ann_f=ann_f, legend=legend)

    return fig

# --- Case studies

def unconstrained(train_set, val_set, N=500, report_step=-1, figure=None):
    log('Unconstrained:', end=' ')

    t0 = time.time()
    baseline_model, baseline_dmp = LTL.create_model_for_train_set(train_set)
    baseline_train_losses, baseline_val_losses = LTL.training_loop(baseline_model, baseline_dmp, train_set, val_set, None, False, nu=1.0, N=N, report_step=report_step)
    runtime(t0)

    if isinstance(figure, Path):
        fig = plot_vs_sample(baseline_model, baseline_dmp, train_set[0], legend=True)
        fig.savefig(figure)
        plt.close(fig=fig)
        fig = plot_vs_sample(baseline_model, baseline_dmp, train_set[0], legend=False)
        fig.savefig(figure.with_name('nolegend_' + figure.name))
        plt.close(fig=fig)

    return baseline_model

def avoid(train_set, val_set, N=500, report_step=-1, figure=None, baseline_model=None):
    log('Avoid:', end=' ')

    avoid_x = 0.4
    avoid_y = 0.4
    distance = 0.1

    avoid_pt = torch.Tensor([avoid_x,avoid_y]).to(DEVICE)

    #define a function that given calculates the distance from the avoid point for any x,y coordinate pair
    dist_f = lambda x : torch.sqrt(torch.sum((x-avoid_pt)**2,dim=-1)).unsqueeze(-1) # Dist from avoid_pt

    #we modify the results graph to include a coloured circle showing the area we want to avoid
    def ann_plt(plt,fig,ax):
        ax.add_artist(plt.Circle((avoid_x, avoid_y), distance, color='r', alpha=0.1))
        plt.scatter([avoid_x],[avoid_y],marker='x',color='r', s=100)

    #define the constraint, including the functions and constants needed to modify the trace to calculate the results we need
    #note in the constraint definition, "1" refers to the distance constant, and "0" refers to the calculated distance from the avoid point
    #thus, the constraint means: ALWAYS (distance constant) <= (distance from avoid point)
    constraint = Constraint("G 1 <= 0", # constraint string
                                dist_f, # distance transform
                                [distance], # distance threshold
                                0.005) #gamma

    #train the model and output time taken every 10 iterations for 500 iterations total; then print the results
    t0 = time.time()
    model, dmp = LTL.create_model_for_train_set(train_set)
    train_losses, val_losses = LTL.training_loop(model, dmp, train_set, val_set, constraint, False, nu=1.0, N=N, report_step=report_step)
    runtime(t0)

    if isinstance(figure, Path) and not baseline_model is None:
        fig = plot_models(baseline_model, 'Unconstrained', model, 'Constrained', dmp, train_set[0], ann_f=ann_plt, legend=True)
        fig.savefig(figure)
        plt.close(fig=fig)
        fig = plot_models(baseline_model, 'Unconstrained', model, 'Constrained', dmp, train_set[0], ann_f=ann_plt, legend=False)
        fig.savefig(figure.with_name('nolegend_' + figure.name))
        plt.close(fig=fig)

def until(train_set, val_set, N=500, report_step=-1, figure=None, baseline_model=None):
    log('Until:', end=' ')

    xthresh = 0.6
    ythresh = 0.4

    #modify the results graph to put red lines at our thresholds
    def ann_plt(plt,fig,ax):
        plt.axhline(ythresh, color='r')
        plt.axvline(xthresh, color='r')

    #define the constraint to be used.
    #"(y <= 0.4) Until (0.6 <= x)"
    constraint = Constraint("(1 <= 2) U (3 <= 0)", # constraint string
                                None, # identity transform
                                [ythresh,xthresh], # consts
                                    0.005) # gamma

    #train the model and output time taken every 10 iterations for 500 iterations total; then print the results
    t0 = time.time()
    model, dmp = LTL.create_model_for_train_set(train_set)
    train_losses, val_losses = LTL.training_loop(model, dmp, train_set, val_set, constraint, False, nu=1.0, N=N, report_step=report_step)
    runtime(t0)

    if isinstance(figure, Path) and not baseline_model is None:
        fig = plot_models(baseline_model, 'Unconstrained', model, 'Constrained', dmp, train_set[0], ann_f=ann_plt, legend=True)
        fig.savefig(figure)
        plt.close(fig=fig)
        fig = plot_models(baseline_model, 'Unconstrained', model, 'Constrained', dmp, train_set[0], ann_f=ann_plt, legend=False)
        fig.savefig(figure.with_name('nolegend_' + figure.name))
        plt.close(fig=fig)

def patrol(train_set, val_set, N=500, report_step=-1, figure=None, baseline_model=None):
    log('Patrol:', end=' ')

    reach_pt1 = torch.Tensor([0.2,0.4]).to(DEVICE)
    reach_pt2 = torch.Tensor([0.85,0.6]).to(DEVICE)

    #define a function d that calculates the distance from the reach points for any x,y coordinate pair
    def dist_tensor(pt):
        return (lambda x : torch.sqrt(torch.sum((x-pt)**2,dim=-1)).unsqueeze(-1))

    d = lambda x : torch.cat((dist_tensor(reach_pt1)(x), dist_tensor(reach_pt2)(x)), dim=-1)

    #modify the results graph to put crosses at each of the reach points
    def ann_plt(plt,fig,ax):
        plt.scatter([0.2],[0.4],marker='x',color='g', s=100)
        plt.scatter([0.85],[0.6],marker='x',color='g', s=100)

    #define the constraint to be used.
    #"(Eventually (distance to pt 1 <= 0.0)) And (Eventually (distance to pt 2 <= 0.0))"
    constraint = Constraint("(F 0 <= 2) & (F 1 <= 3)", # constraint string
                                d, # distance transform
                                [0.0,0.0], # distance thresholds to pt1 and pt2, respectively
                                0.005) # gamma

    #train the model and output time taken every 10 iterations for 500 iterations total; then print the results
    t0 = time.time()
    model, dmp = LTL.create_model_for_train_set(train_set)
    train_losses, val_losses = LTL.training_loop(model, dmp, train_set, val_set, constraint, False, nu=1.0, N=N, report_step=report_step)
    runtime(t0)

    if isinstance(figure, Path) and not baseline_model is None:
        fig = plot_models(baseline_model, 'Unconstrained', model, 'Constrained', dmp, train_set[0], ann_f=ann_plt, legend=True)
        fig.savefig(figure)
        plt.close(fig=fig)
        fig = plot_models(baseline_model, 'Unconstrained', model, 'Constrained', dmp, train_set[0], ann_f=ann_plt, legend=False)
        fig.savefig(figure.with_name('nolegend_' + figure.name))
        plt.close(fig=fig)

def loop(train_set, val_set, N=500, report_step=-1, figure=None, baseline_model=None):
    log('Loop:', end=' ')

    #define the points to be reached
    reach_pt1 = torch.Tensor([0.2,0.6]).to(DEVICE)
    reach_pt2 = torch.Tensor([0.4,0.8]).to(DEVICE)

    #define a function d that calculates the distance from the reach points for any x,y coordinate pair
    def dist_tensor(pt):
        return (lambda x : torch.sqrt(torch.sum((x-pt)**2,dim=-1)).unsqueeze(-1))

    d = lambda x : torch.cat((dist_tensor(reach_pt1)(x), dist_tensor(reach_pt2)(x)), dim=-1)

    #modify the results graph to put crosses at each of the reach points
    def ann_plt(plt,fig,ax):
        plt.scatter([0.2],[0.6],marker='x',color='g', s=100)
        plt.scatter([0.4],[0.8],marker='x',color='g', s=100)

    #define the constraint to be used.
    #"Eventually ((distance to pt 1 <= 0.0) And (Weak Next (Eventually (distance to pt 2 <= 0.0))))"
    constraint = Constraint("F ((1 <= 2) & (X (F (0 <= 3))))", # constraint string
                                d, # distance transform
                                [0.0,0.0], # distance thresholds to pt1 and pt2, respectively
                                0.005) # gamma

    #train the model and output time taken every 10 iterations for 500 iterations total; then print the results
    t0 = time.time()
    model, dmp = LTL.create_model_for_train_set(train_set)
    train_losses, val_losses = LTL.training_loop(model, dmp, train_set, val_set, constraint, False, nu=1.0, N=N, report_step=report_step)
    runtime(t0)

    if isinstance(figure, Path) and not baseline_model is None:
        fig = plot_models(baseline_model, 'Unconstrained', model, 'Constrained', dmp, train_set[0], ann_f=ann_plt, legend=True)
        fig.savefig(figure)
        plt.close(fig=fig)
        fig = plot_models(baseline_model, 'Unconstrained', model, 'Constrained', dmp, train_set[0], ann_f=ann_plt, legend=False)
        fig.savefig(figure.with_name('nolegend_' + figure.name))
        plt.close(fig=fig)

def compound(train_set, val_set, N=500, report_step=-1, figure=None, baseline_model=None):
    log('Compound:', end=' ')

    #define the points to be reached and avoided
    avoid_pt = torch.Tensor([0.5,0.5]).to(DEVICE)
    reach_pt = torch.Tensor([0.7,0.5]).to(DEVICE)
    dist = 0.1
    y_thresh = 0.8

    #define a function d that calculates the distance from the reach and avoid points for any x,y coordinate pair, and its y component
    def dist_tensor(pt):
        return (lambda x : torch.sqrt(torch.sum((x-pt)**2,dim=-1)).unsqueeze(-1))

    def y_tensor():
        return (lambda x : x[:,:,1].unsqueeze(-1))

    d = lambda x : torch.cat((dist_tensor(avoid_pt)(x), dist_tensor(reach_pt)(x), y_tensor()(x)), dim=-1)

    #modify the results graph to put crosses at each of the reach points, add a red avoidance circle, and put a red line at the y threshold
    def ann_plt(plt,fig,ax):
        ax.add_artist(plt.Circle((0.5, 0.5), dist, color='r', alpha=0.1))
        plt.scatter([0.5],[0.5],marker='x',color='r', s=100)
        plt.scatter([0.7],[0.5],marker='x',color='r', s=100)
        plt.axhline(y_thresh, color='r')

    #define the constraint to be used.
    #"(Always (0.1 <= distance from avoid point)) And (Eventually (distance from reach point <= 0.0)) And (Always (y-coordinate <= 0.8))"
    constraint = Constraint("(G (3 <= 0)) & (F (1 <= 4)) & (G (2 <= 5))", # constraint string
                                d, # transform
                                [dist,0.0,y_thresh], # distance thresholds to pt1 and pt2, respectively, and const for y
                                0.005) # gamma

    #train the model and output time taken every 10 iterations for 500 iterations total; then print the results
    t0 = time.time()
    model, dmp = LTL.create_model_for_train_set(train_set)
    train_losses, val_losses = LTL.training_loop(model, dmp, train_set, val_set, constraint, False, nu=1.0, N=N, report_step=report_step)
    runtime(t0)

    if isinstance(figure, Path) and not baseline_model is None:
        fig = plot_models(baseline_model, 'Unconstrained', model, 'Constrained', dmp, train_set[0], ann_f=ann_plt, legend=True)
        fig.savefig(figure)
        plt.close(fig=fig)
        fig = plot_models(baseline_model, 'Unconstrained', model, 'Constrained', dmp, train_set[0], ann_f=ann_plt, legend=False)
        fig.savefig(figure.with_name('nolegend_' + figure.name))
        plt.close(fig=fig)

def double_conj_dmp(train_set, val_set, N=500, report_step=-1, figure=None, baseline_model=None):
    log('Double loop (conjoined, DMP):', end=' ')

    #define the points to be reached
    reach_pt1 = torch.Tensor([0.2,0.6]).to(DEVICE)
    reach_pt2 = torch.Tensor([0.4,0.8]).to(DEVICE)
    reach_pt3 = torch.Tensor([0.6,0.2]).to(DEVICE)
    reach_pt4 = torch.Tensor([0.8,0.4]).to(DEVICE)

    #define a function d that calculates the distance from the reach points for any x,y coordinate pair
    def dist_tensor(pt):
        return (lambda x : torch.sqrt(torch.sum((x-pt)**2,dim=-1)).unsqueeze(-1))

    d = lambda x : torch.cat((dist_tensor(reach_pt1)(x),
                             dist_tensor(reach_pt2)(x),
                             dist_tensor(reach_pt3)(x),
                             dist_tensor(reach_pt4)(x)),
                             dim=-1)

    #modify the results graph to put crosses at each of the reach points
    def ann_plt(plt,fig,ax):
            plt.scatter([0.2],[0.6],marker='x',color='g', s=100)
            plt.scatter([0.4],[0.8],marker='x',color='g', s=100)
            plt.scatter([0.6],[0.2],marker='x',color='g', s=100)
            plt.scatter([0.8],[0.4],marker='x',color='g', s=100)

    #define the constraint to be used.
    #"(Eventually ((distance from reach point 2 <= 0.0) And (Weak Next (Eventually (distance from reach point 1 <= 0.0)))))
    #  And (Eventually ((distance from reach point 4 <= 0.0) And (Weak Next (Eventually (distance from reach point 3 <= 0.0)))))"
    c_d1 = "0 <= 4"
    c_d2 = "1 <= 5"
    c_d3 = "2 <= 6"
    c_d4 = "3 <= 7"

    ltl_constraint = "F ({c_d2} & X (F {c_d1})) & F({c_d4} & X(F {c_d3}))".format(c_d1=c_d1, c_d2=c_d2, c_d3 = c_d3, c_d4 = c_d4)

    constraint = Constraint(ltl_constraint, # constraint string
                            d, # distance transform
                            [0.0,0.0,0.0,0.0], # distance thresholds to pt1 and pt2, respectively
                             0.01) # gamma

    #train the model and output time taken every 10 iterations for 500 iterations total; then print the results
    t0 = time.time()
    model, dmp = LTL.create_model_for_train_set(train_set)
    train_losses, val_losses = LTL.training_loop(model, dmp, train_set, val_set, constraint, False, nu=1.0, N=N, report_step=report_step)
    runtime(t0)

    if isinstance(figure, Path) and not baseline_model is None:
        fig = plot_models(baseline_model, 'Unconstrained', model, 'Constrained', dmp, train_set[0], ann_f=ann_plt, legend=True)
        fig.savefig(figure)
        plt.close(fig=fig)
        fig = plot_models(baseline_model, 'Unconstrained', model, 'Constrained', dmp, train_set[0], ann_f=ann_plt, legend=False)
        fig.savefig(figure.with_name('nolegend_' + figure.name))
        plt.close(fig=fig)

def direct_pos(rand_gen=None, N=500, figure=None):
    log('Direct - positions:', end=' ')

    exectype = 'ocaml'
    dt = 1
    d = 2
    init_line = {
        'start': (0.,)*d, 'stop': (1.,)*d, 'num': 50,
        'dt': dt, 'rand': 0.005, 'come_back': False
    }
    lr = 0.1                # learning rate
    smoothing_parameter=0.05

    # point p to avoid (centre of the circle)
    avoid_x = 0.4
    avoid_y = 0.4
    avoid_pt = torch.Tensor([avoid_x, avoid_y]).to(DEVICE)

    avoid_radius = 0.1
    speed_limit = 0.15 / dt
    acceleration_limit = 0.02 / (dt**2)

    constraint = Constraint(
        (                       # constraint string, indexes into post-transform state:
            "G ( (3 <= 0) & " + # distance to point always greater than const
                "(1 <= 4) & " + # total speed always less than const
                "(2 <= 5) )"    # total acceleration always less than const
        ),
        lambda x : torch.cat(
            [   # first part of post-transform state:
                x.distance_from(avoid_pt, in_constraint=True),
                x.total_speed(True),
                x.total_acceleration(True)
            ],
            dim=-1
        ),
        [                               # second part of post-transform state:
            avoid_radius,               # 3: distance threshold to point
            speed_limit,                # 4: speed limit
            acceleration_limit,         # 5: acceleration limit
        ],
        smoothing_parameter,
        exectype
    )

    t0 = time.time()

    Z = FixedEndpointTrajectory.new_line(**init_line, rand_gen=rand_gen)
    Zs = [Z.detach_clone()]
    losses = [constraint.loss(Z).detach().numpy()]

    optimizer = optim.SGD(list(Z.parameters()), lr=lr)

    def train(Z):
        # get LTL loss (trajectory as: batch x time x state)
        loss = constraint.loss(Z)

        # backward pass and update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss

    for i in range(N):
        loss = train(Z)
        losses.append(loss.detach().numpy())
        Zs.append(Z.detach_clone())
        if torch.any(torch.isnan(Z.points)):
            break_msg = ""
            if torch.all(torch.isnan(Z.points)):
                break_msg = " ALL ELEMENTS"
            else:
                break_msg = "\n%s"%(torch.isnan(Z.points))
            log("Z has NaN elements at:" + break_msg)
            break
        if loss <= 5e-3:
            log("Stopped at iteration %i"%(i))
            break

    runtime(t0)

    if isinstance(figure, Path):
        def ann_f(plt,fig,ax):
            ax.add_artist(plt.Circle((avoid_x, avoid_y), avoid_radius, color='r', alpha=0.1))
            ax.scatter([avoid_x],[avoid_y],marker='x',color='r', s=100)
        (fig, ax) = plot_two_trajectories(Zs[0].get_position().detach().cpu(), "Initial", Zs[-1].get_position().detach().cpu(), "Final", size=(8,8), ann_f=ann_f, legend=True)
        fig.savefig(figure)
        plt.close(fig=fig)
        (fig, ax) = plot_two_trajectories(Zs[0].get_position().detach().cpu(), "Initial", Zs[-1].get_position().detach().cpu(), "Final", size=(8,8), ann_f=ann_f, legend=False)
        fig.savefig(figure.with_name('nolegend_' + figure.name))
        plt.close(fig=fig)

def direct_pos_vel(rand_gen=None, N=500, figure=None):
    log('Direct - position+velocity:')

    exectype = 'ocaml'
    dt = 1
    d = 2
    base_iter = int(N / 4)
    ltl_iter = N - base_iter
    lr = 0.1                # learning rate
    smoothing_parameter=0.01 # Note: different from position-only case study
    weight_ltl = 5

    speed_limit = 0.03 / dt
    acceleration_limit = 0.001 / (dt**2)

    endpoints = (
        torch.Tensor((0.,)*d),
        torch.Tensor((1.,)*d)
    )

    constraint = Constraint(    # upper and lower speed limits
        "G ( (0 >= 2) & (1 >= 2) & (0 <= 3) & (1 <= 3) )",
        lambda x: x.get_speed().unsqueeze(0),
        [0.0, speed_limit],
        smoothing_parameter,
        exectype
    )

    constraints = [constraint]

    Z = PhaseSpaceTrajectory.new_random(50, d, dt, low=endpoints[0][0], high=endpoints[1][0], rand_gen=rand_gen)

    Zs = [Z.detach_clone()]
    Zs_ltl = [Z.detach_clone()]
    losses = []
    losses_ltl= []

    log('-> base training:', end=' ')

    t0 = time.time()

    optimizer = optim.SGD(list(Z.parameters()), lr=lr)

    def train(Z, weight_ltl=None):
        dyn_loss = Z.get_dynamic_loss(fix_endpoints=endpoints)

        if weight_ltl is not None:
            # get LTL loss (trajectory as: batch x time x state)
            ltl_losses = [c.loss(Z) for c in constraints]
            ltl_loss = sum(ltl_losses)
            loss = dyn_loss + weight_ltl * ltl_loss
            out_loss = [dyn_loss.detach()]
            out_loss.extend([l.detach() for l in ltl_losses])
        else:
            loss = dyn_loss
            out_loss = dyn_loss.detach()

        # backward pass and update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return out_loss

    for i in range(base_iter):
        loss = train(Z)
        losses.append(loss)
        Zs.append(Z.detach_clone())
        if torch.any(torch.isnan(Z.points)):
            if torch.all(torch.isnan(Z.points)):
                break_msg = " ALL ELEMENTS"
            else:
                break_msg = "\n%s"%(torch.isnan(Z.points))
            log("Z has NaN elements at:" + break_msg)
            break
        if (loss.detach().numpy() <= 1e-10):
            log("Stopped at iteration %i"%(i))
            break

    runtime(t0)

    log('-> LTL training:', end=' ')

    t0 = time.time()

    Z = Zs_ltl[0].detach_clone()
    optimizer = optim.SGD(list(Z.parameters()), lr=lr)

    for i in range(ltl_iter):
        ls = train(Z, weight_ltl=weight_ltl)
        dyn = ls[0]
        ltls = ls[1:]
        losses_ltl.append(ls)
        loss = dyn + weight_ltl*sum(ltls)
        Zs_ltl.append(Z.detach_clone())
        if torch.any(torch.isnan(Z.points)):
            if torch.all(torch.isnan(Z.points)):
                break_msg = " ALL ELEMENTS"
            else:
                break_msg = "\n%s"%(torch.isnan(Z.points))
            log("Z has NaN elements at:" + break_msg)
            assert False
        if (loss.detach().numpy() <= 3e-2):
            log("Stopped at iteration %i"%(i))
            break

    runtime(t0)

    if isinstance(figure, Path):
        (fig, ax) = plot_two_trajectories(Zs[-1].points.detach().cpu(), "Dynamical only", Zs_ltl[-1].points.detach().cpu(), "Dynamical and LTL", size=(8,8), legend=True)
        fig.savefig(figure)
        plt.close(fig=fig)
        (fig, ax) = plot_two_trajectories(Zs[-1].points.detach().cpu(), "Dynamical only", Zs_ltl[-1].points.detach().cpu(), "Dynamical and LTL", size=(8,8), legend=False)
        fig.savefig(figure.with_name('nolegend_' + figure.name))
        plt.close(fig=fig)

def direct_double_conj(rand_gen=None, N=500, figure=None):
    log('Direct - double loop conjoined:', end=' ')

    exectype = 'ocaml'
    dt = 1
    d = 2
    lr = 0.05                # Note: not the same for all case studies
    smoothing_parameter=0.05 # Note: not the same for all case studies
    dynamics_weight = 0.1

    speed_limit = 0.05 / dt
    until_radius = 0.0

    init_default = {
        "start": (0.,)*d, "stop": (1.,)*d, "num": 50,
        "dt": dt, "rand": 0.0025, "come_back": False
    }

    reach_points = [
        torch.Tensor(p).to(DEVICE)
        for p in [[0.2,0.6], [0.4,0.8], [0.6,0.2], [0.8,0.4]]
    ]
    reach_Points = [Point.from_tensor(rp) for rp in reach_points]

    conjoined_loop = PhaseSpaceTrajectory.double_loop_conjoined(
        speed_limit=speed_limit,
        points=reach_Points,
        reach_margin=until_radius,
        smoothing_parameter=smoothing_parameter
    )

    # Note: notebook has imitation loss, but its weight is 0
    def train(Z, dynamics=dynamics_weight):
        # get LTL loss
        loop_loss = conjoined_loop.loss(Z)
        dyn_loss = Z.get_dynamic_loss(fix_endpoints=(p1,p2))

        loss = loop_loss + dynamics*dyn_loss

        # backward pass and update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item(), loop_loss.item(), dyn_loss.item()

    Z = PhaseSpaceTrajectory.new_line(**init_default, rand_gen=rand_gen)

    z = Z.detach_clone()
    Zs = [z]
    p1 = torch.Tensor(init_default["start"])
    p2 = torch.Tensor(init_default["stop"])
    with torch.no_grad():
        ltl = conjoined_loop.loss(Z).item()
        dyn = Z.get_dynamic_loss(fix_endpoints=(p1,p2)).item()
    losses = [(ltl + dynamics_weight*dyn, ltl, dyn)]

    optimizer = optim.SGD(list(Z.parameters()), lr=lr)

    t0 = time.time()
    for i in range(N):
        loss, loop, dyn = train(Z, dynamics=dynamics_weight)
        losses.append([loss, loop, dyn])
        z = Z.detach_clone()
        Zs.append(z)
    runtime(t0)

    if isinstance(figure, Path):
        #modify the results graph to put crosses at each of the reach points
        def ann_plt(plt,fig,ax):
            plt.scatter([0.2],[0.6],marker='x',color='g', s=100)
            plt.scatter([0.4],[0.8],marker='x',color='g', s=100)
            plt.scatter([0.6],[0.2],marker='x',color='g', s=100)
            plt.scatter([0.8],[0.4],marker='x',color='g', s=100)
        (fig, ax) = plot_one_trajectory(Zs[-1].points.detach().cpu(), 'Double loop',  size=(8,8), ann_f=ann_plt, legend=True)
        fig.savefig(figure)
        plt.close(fig=fig)
        (fig, ax) = plot_one_trajectory(Zs[-1].points.detach().cpu(), 'Double loop',  size=(8,8), ann_f=ann_plt, legend=False)
        fig.savefig(figure.with_name('nolegend_' + figure.name))
        plt.close(fig=fig)

def direct_double_nest(rand_gen=None, N=500, figure=None):
    log('Direct - double loop nested:', end=' ')

    exectype = 'ocaml'
    dt = 1
    d = 2
    lr = 0.05                # Note: not the same for all case studies
    smoothing_parameter=0.01 # Note: not the same for all case studies
    dynamics_weight = 0.1

    speed_limit = 0.05 / dt
    until_radius = 0.0

    init_default = {
        "start": (0.,)*d, "stop": (1.,)*d, "num": 50,
        "dt": dt, "rand": 0.0025, "come_back": False
    }

    reach_points = [
        torch.Tensor(p).to(DEVICE)
        for p in [[0.2,0.6], [0.4,0.8], [0.6,0.2], [0.8,0.4]]
    ]
    reach_Points = [Point.from_tensor(rp) for rp in reach_points]

    nested_loop = PhaseSpaceTrajectory.double_loop_nested(
        speed_limit=speed_limit,
        points=reach_Points,
        reach_margin=until_radius,
        smoothing_parameter=smoothing_parameter
    )

    # Note: notebook has imitation loss, but its weight is 0
    def train(Z, dynamics=dynamics_weight):
        # get LTL loss
        loop_loss = nested_loop.loss(Z)
        dyn_loss = Z.get_dynamic_loss(fix_endpoints=(p1,p2))

        loss = loop_loss + dynamics*dyn_loss

        # backward pass and update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item(), loop_loss.item(), dyn_loss.item()

    Z = PhaseSpaceTrajectory.new_line(**init_default, rand_gen=rand_gen)

    z = Z.detach_clone()
    Zs = [z]
    p1 = torch.Tensor(init_default["start"])
    p2 = torch.Tensor(init_default["stop"])
    with torch.no_grad():
        ltl = nested_loop.loss(Z).item()
        dyn = Z.get_dynamic_loss(fix_endpoints=(p1,p2)).item()
    losses = [(ltl + dynamics_weight*dyn, ltl, dyn)]

    optimizer = optim.SGD(list(Z.parameters()), lr=lr)

    t0 = time.time()
    for i in range(N):
        loss, loop, dyn = train(Z, dynamics=dynamics_weight)
        losses.append([loss, loop, dyn])
        z = Z.detach_clone()
        Zs.append(z)
    runtime(t0)

    if isinstance(figure, Path):
        #modify the results graph to put crosses at each of the reach points
        def ann_plt(plt,fig,ax):
            plt.scatter([0.2],[0.6],marker='x',color='g', s=100)
            plt.scatter([0.4],[0.8],marker='x',color='g', s=100)
            plt.scatter([0.6],[0.2],marker='x',color='g', s=100)
            plt.scatter([0.8],[0.4],marker='x',color='g', s=100)
        (fig, ax) = plot_one_trajectory(Zs[-1].points.detach().cpu(), 'Double loop',  size=(8,8), ann_f=ann_plt, legend=True)
        fig.savefig(figure)
        plt.close(fig=fig)
        (fig, ax) = plot_one_trajectory(Zs[-1].points.detach().cpu(), 'Double loop',  size=(8,8), ann_f=ann_plt, legend=False)
        fig.savefig(figure.with_name('nolegend_' + figure.name))
        plt.close(fig=fig)

# --- Execution

# Number of times to run each case study
REPEATS = 2

def randomise_seeds():
    random_data = os.urandom(4)
    seed = int.from_bytes(random_data, sys.byteorder)

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    G = np.random.default_rng(seed=seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return (seed, G)

def divider_run():
    log('='*10)

def divider_sec():
    log('-'*10)

def pad_to_tens(i):
    if i < 10:
        return '0' + str(i)
    else:
        return str(i)

if __name__ == '__main__':
    # Load data
    dmp_train, dmp_val = LTL.load_data("drive/demos/patrol")

    # Prep figure directory
    FIGS.mkdir(parents=True, exist_ok=True)

    # Run through each case study as many times as configured
    for i in range(0, REPEATS):
        divider_run()
        log('Run number: {}'.format(i))
        (seed, G) = randomise_seeds()
        log('Seeds set to: {}'.format(seed))
        divider_sec()
        baseline_model = unconstrained(dmp_train, dmp_val, N=500, report_step=-1, figure=FIGS / 'unconstrained-{}.png'.format(pad_to_tens(i)))
        avoid(dmp_train, dmp_val, N=500, report_step=-1, figure=FIGS / 'avoid-{}.png'.format(pad_to_tens(i)), baseline_model=baseline_model)
        until(dmp_train, dmp_val, N=500, report_step=-1, figure=FIGS / 'until-{}.png'.format(pad_to_tens(i)), baseline_model=baseline_model)
        patrol(dmp_train, dmp_val, N=500, report_step=-1, figure=FIGS / 'patrol-{}.png'.format(pad_to_tens(i)), baseline_model=baseline_model)
        loop(dmp_train, dmp_val, N=500, report_step=-1, figure=FIGS / 'loop-{}.png'.format(pad_to_tens(i)), baseline_model=baseline_model)
        compound(dmp_train, dmp_val, N=500, report_step=-1, figure=FIGS / 'compound-{}.png'.format(pad_to_tens(i)), baseline_model=baseline_model)
        direct_pos(rand_gen=G, N=600, figure=FIGS / 'direct_pos-{}.png'.format(pad_to_tens(i)))
        direct_pos_vel(rand_gen=G, N=400, figure=FIGS / 'direct_pos_vel-{}.png'.format(pad_to_tens(i)))
        double_conj_dmp(dmp_train, dmp_val, N=500, report_step=-1, figure=FIGS / 'double-conj-dmp-{}.png'.format(pad_to_tens(i)), baseline_model=baseline_model)
        direct_double_conj(rand_gen=G, N=1000, figure=FIGS / 'direct_double_conj-{}.png'.format(pad_to_tens(i)))
        direct_double_nest(rand_gen=G, N=80, figure=FIGS / 'direct_double_nest-{}.png'.format(pad_to_tens(i)))
