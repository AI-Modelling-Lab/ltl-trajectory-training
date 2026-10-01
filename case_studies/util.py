import matplotlib.pyplot as plt
import torch
import time
from datetime import datetime


### General ###
###############

def t_stamp():
    return datetime.now().strftime("%Y-%m-%d-%H")

def runtime(t0):
    diff = time.time()-t0
    print("Runtime : {} hr {} min {} sec".format(int(diff//3600),int((diff%3600)//60), int(diff%60)))

def L2(components_tensor):
    """
    Compute the L2 (Euclidean) norm of a tensor whose final axis
    is the state component dimension. E.g. a trajectory (n,d) with n sample
    points and d coordinates per point.
    """
    return torch.sqrt(torch.sum(components_tensor**2, dim=-1))



### Plotting ###
################

def get_shared_xaxis(ylim=[-0.6,2.0]):  # xlim?
    fig, (ax_pos, ax_vel) = plt.subplots(
        nrows=2, sharex=True,
        gridspec_kw={'height_ratios': [6, 1], 'hspace': 0},
        # subplot_kw={'frameon': False},
    )

    # trajectory subplot settings
    ax_pos.set_ylabel("y-coordinate ", loc="top")
    ax_pos.axis("equal")
    ax_pos.set_ylim(ylim)
    # ax_pos.legend(loc="upper left")

    # velocity subplot settings
    # ax_vel.legend(loc="upper left")
    ax_vel.yaxis.set_label_position("right")
    ax_vel.yaxis.tick_right()
    ax_vel.set_ylabel(" velocity of final path", loc="bottom")

    plt.xlabel("x-coordinate")
    plt.rcParams.update({'font.size': 12})
    plt.tight_layout()

    return fig, (ax_pos, ax_vel)    # keep the (. (. .)) shape for back-compat

def get_split_xaxis(ylim=[-1.0,1.5]):  # xlim?
    fig, (ax_pos, ax_vel) = plt.subplots(
        nrows=2, sharex=False,
        gridspec_kw={'height_ratios': [6, 1], 'hspace': 0.25},
        # subplot_kw={'frameon': False},
    )

    # trajectory subplot settings
    ax_pos.set_ylabel("y-coordinate ", loc="top")
    ax_pos.axis("equal")
    ax_pos.set_ylim(ylim)
    ax_pos.legend(loc="upper left")

    # velocity subplot settings
    ax_vel.legend(loc="upper left")
    ax_vel.yaxis.set_label_position("right")
    ax_vel.yaxis.tick_right()
    ax_vel.set_ylabel(" velocity of final path", loc="bottom")

    plt.xlabel("x-coordinate")
    plt.rcParams.update({'font.size': 12})
    plt.tight_layout()

    return fig, (ax_pos, ax_vel)    # keep the (. (. .)) shape for back-compat

def ann_plt(ax, reach_points):
    reach_centre = torch.sum(torch.stack(reach_points, dim=0), dim=0) / len(reach_points)
    def offset(pt, c=reach_centre, w=0.2):
        # return pt, but offset by some small distance away from c
        return pt + w*(pt-c)
    def mark(point, label):
        ax.scatter(point[0], point[1], marker='x', color='g', s=100)
        op = offset(point)
        ax.text(op[0], op[1], label, ha="center")
    for i,p in enumerate(reach_points):
        mark(p, f"$o_{i+1}$")

def double_loop_plot(
    ax_pos, Zs, reach_points,
    arrow_threshold = 0.2,
    lims=[-0.1,1.1],
    plot_initial=False,
    plot_intermediate=None,
    axis_labels=True,
    arrow_idx_offset = 5,
):
    # plot the points we're looping to
    ann_plt(ax_pos, reach_points)

    # plot initial trajectory
    if plot_initial:
        ixy = Zs[0].points.detach()
        ax_pos.plot(ixy[:,0], ixy[:,1], label="Initial", linestyle='dashed', c="black")

    # plot final (learnt/constrained) trajectory
    xys = Zs[-1].points.detach()
    ax_pos.plot(xys[:,0], xys[:,1], label="Final", c="orange", linewidth=2)

    # plot intermediate trajectory
    if plot_intermediate is not None:
        intermediate_indices = set([len(Zs)//i for i in plot_intermediate])
        for i in intermediate_indices:
            xys = Zs[i].points.detach()
            ax_pos.plot(xys[:,0], xys[:,1], label=f"Iteration {i}", linewidth=1, linestyle="dotted")

    # trajectory subplot settings
    if axis_labels:
        ax_pos.set_xlabel("x-coordinate")
        ax_pos.set_ylabel("y-coordinate")
    ax_pos.set_aspect('equal', adjustable='box')
    ax_pos.set_ylim(lims)
    ax_pos.set_xlim(lims)
    ax_pos.legend(loc="upper left")

    # finally, draw arrows if required
    # TODO do this for intermediate trajectories?
    if arrow_threshold:
        # find timesteps with large displacement (for visual clarity, don't draw all arrows)
        disp = xys[1:]-xys[:-1]
        jump_ids = torch.nonzero(L2(disp) > arrow_threshold).flatten()
        # add a single arrow before/after the double loop (before/after lowest/largest index found so far)
        jump_ids = torch.cat([
            jump_ids,
            (torch.min(jump_ids) - arrow_idx_offset).flatten(),
            (torch.max(jump_ids) + arrow_idx_offset).flatten()
        ])
        # coordinates and directions for arrows
        arrow_coords = xys[jump_ids,:] + 0.34*(disp[jump_ids,:])
        assert arrow_coords.shape[0] == jump_ids.shape[0]
        assert arrow_coords.shape[1] == xys.shape[1]
        arrow_dirs = disp[jump_ids,:]
        # normalise arrow size
        arrow_dirs = arrow_dirs / L2(arrow_dirs).unsqueeze(-1)
        # draw the quiver
        ax_pos.quiver(arrow_coords[:,0], arrow_coords[:,1], arrow_dirs[:,0], arrow_dirs[:,1],
            angles='xy', scale_units='xy', scale=10,
            headwidth=5, color="orange",
        )

def plot_trajectory_and_speed(initial, final, labels=None, ylim=[-1,1]):
    """
    Plotting function: just hand it the initial and final states.
    For the mixed optimisation experiment.
    """
    fig, (ax_pos, ax_vel) = get_shared_xaxis(ylim=ylim)

    if labels is None:
        labels=["Initial", "Final"]

    # plot initial trajectory
    ixy = initial.points.detach()
    ax_pos.plot(ixy[:,0], ixy[:,1], label=labels[0], linestyle='dashed')

    # plot final (learnt/constrained) trajectory
    xys = final.points.detach()
    ax_pos.plot(xys[:,0], xys[:,1], label=labels[1], linewidth=3)

    # plot velocity
    us = final.get_speed().detach()
    ax_vel.plot(xys[1:,0], us[:,0], label="x-speed", c='black', linestyle="dashed")
    ax_vel.plot(xys[1:,0], us[:,1], label="y-speed", c='g')

    ax_pos.legend(loc="upper left")
    ax_vel.legend(loc="upper left")

    return fig, ax_pos, ax_vel