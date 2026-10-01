# from datetime import datetime
# import time
import numbers

import numpy as np
import matplotlib.pyplot as plt
import torch
from util import L2
# from torch.autograd import Function
# import torch.autograd as autograd
# import torch.optim as optim

from ltl.constraint import Constraint

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")



### Trajectories ###
####################

# def init_line(start, stop, num, come_back=False, rand=None):
#     """
#     Generates a set of `num` points evenly spaced along the line
#     from `start` to `stop` (both included).

#     Args:
#         `start`: `D`-dimensional `np.array` or array-like
#         `stop`: `D`-dimensional `np.array` or array-like
#     Returns:
#         a numpy array of shape `(num,D)` (or `(2*num-1,D)` if `come_back`)
#     """
#     xy, dx = np.linspace(start, stop, num, retstep=True)
#     if come_back:
#         # concatenate a flipped trajectory (with the turning point duplicate removed)
#         xy = np.concatenate((xy, np.flip(xy, axis=0)[1:]), axis=0)
#         assert (xy[0]==xy[-1]).all(), "First and last points of trajectory should be equal:\nx: {} = {}\ny: {} = {}".format(xy[0,0], xy[-1,0], xy[0,1], xy[-1,1])
#     if rand is not None:
#         xy = xy + np.random.default_rng().normal(size=xy.shape, scale=rand)
#     return xy



class Trajectory(torch.nn.Module):
    """
    A list of points interpreted as position samples from a trajectory.
    """
    def __init__(self, points: torch.Tensor, timesteps=0.1):
        """
        Args:
            points: Trajectory, represented as a tensor of `N` points. Its shape is
                `(N, D)`, where `D` is the dimension of the space.
            timesteps: Can be a number or an array of numbers of shape `(N-1,)`.
                If it is a number, it is interpreted as the constant time
                interval between any two successive elements in `points`.
                If it has shape `(N-1,)`, it is interpreted as timesteps between
                successive elements of `points`.
                Default `0.1`.
        """
        # TODO consider setting default timestep to 1
        super().__init__()
        assert isinstance(timesteps, numbers.Number) or (points.shape[0]==timesteps.shape[0]+1)
        # assert (
        #     isinstance(timesteps, numbers.Number)
        # ) or (
        #     (points.shape[0]==timesteps.shape[0]+1) and isinstance(timesteps[0], numbers.Number) and (len(timesteps.shape) == 1)
        # )
        self.points = torch.nn.Parameter(points)
        self._times = timesteps

    @property
    def _dt_cst(self):
        return isinstance(self._times, numbers.Number)

    def _handle_timesteps_input(self, timesteps):
        n = self.points.shape[0]
        if isinstance(timesteps, numbers.Number):
            return torch.Tensor([timesteps] * (n-1))
        else:
            assert (timesteps.shape[0] == n-1)
            return timesteps

    @property
    def timesteps(self):
        return self._handle_timesteps_input(self._times)
    
    def forward(self):
        # The points alone represent the trajectory.
        # Timesteps are used for speed and similar computations only.
        # NOTE One could complicate the model to interpolate using input timestamps.
        return self.points

    def get_displacement(self):
        """
        Compute the displacement vectors between consecutive points.

        Returns:
            tensor of shape `(n-1, D)`, where `self.points.shape = (n, D)`
        """
        return (self.points[1:]-self.points[:-1])

    def get_speed(self, override_timesteps=None, prepend=False):
        """
        Compute the discrete velocities of the trajectory.

        Args:
            override_timesteps: Optional. If given, use this/these timestep/s instead
                of the timesteps saved in this instance.

        Returns:
            tensor of shape `(n-1, D)`, where `self.points.shape = (n, D)`
        """
        # I think torch division does reasonable things with scalars as well.
        # However, the commented lines contain type/shape assertions,
        # at the potential cost of extra memory allocations.
        if override_timesteps:
            # timesteps = self._handle_timesteps_input(override_timesteps)
            timesteps = override_timesteps
        else:
            # timesteps = self.timesteps
            timesteps = self._times
        s = self.get_displacement() / timesteps
        if prepend:
            z = torch.tensor([[0.0]*self.points.shape[1]])
            s = torch.cat([z, s])
        return s

    def get_acceleration(self, override_timesteps=None, prepend=False):
        """
        Compute the discrete accelerations of the trajectory.

        Args:
            override_timesteps: Optional. If given, use this/these timestep/s instead
                of the timesteps saved in this instance.

        Returns:
            tensor of shape `(n-2, D)`, where `self.points.shape = (n, D)`
        """
        v = self.get_speed(override_timesteps, prepend)
        if override_timesteps:
            # timesteps = self._handle_timesteps_input(override_timesteps)
            timesteps = override_timesteps
        else:
            # timesteps = self.timesteps
            timesteps = self._times
        a = (v[1:]-v[:-1]) / timesteps
        if prepend:
            z = torch.tensor([[0.0]*self.points.shape[1]])
            a = torch.cat([z, a])
        return a

    @classmethod
    def new_line(cls, start, stop, num, dt, rand=None, rand_gen=None, come_back=False):
        """
        Generates a trajectory with `num` points evenly spaced along the line
        from `start` to `stop` (both included), with optional randomisation and
        return journey.

        Args:
            start: `D`-dimensional point (tensor, tuple, array, or similar)
            stop: `D`-dimensional point (tensor, tuple, array, or similar)
            rand: size of the added randomisation [None]
            rand_gen: generator for the random component, optional
            come_back: if True, make a trajectory that returns along the same line
        Returns:
            trajectory of shape `(num,D)` (or `(2*num-1,D)` if `come_back`)
        """
        def tensorify(obj):
            if isinstance(obj, torch.Tensor):
                return obj
            return torch.Tensor(obj).to(DEVICE)
        p1 = tensorify(start)
        p2 = tensorify(stop)
        # NOTE torch.linspace only supports one-dimensional start/end
        xy = torch.Tensor(np.linspace(p1.cpu(),p2.cpu(),num=num)).to(DEVICE)
        if come_back:
            # concatenate a flipped trajectory (with the turning point duplicate removed)
            # NOTE torch.flip is slower than np.flip - probably not important,
            # given this method is for initialisation
            xy = torch.concat((xy, torch.flip(xy, dims=(0,))[1:]), dim=0)
        if rand is not None:
            # ensure datatype matches
            R = np.random.default_rng(rand_gen).normal(size=xy.shape, scale=rand)
            xy = xy + torch.Tensor(R).to(DEVICE)
        return cls(xy, dt)

    def detach_clone(self):
        if self._dt_cst:
            t = self._times
        else:
            t = self._times.clone()
        return Trajectory(self.points.detach().clone(), t)



class PhaseSpaceTrajectory(Trajectory):
    """
    A trajectory represented by a list of points and corresponding speeds.
    These must be constrained during training to be useful, physically
    interpretable velocities of the trajectory. See `get_dynamic_loss`.
    """

    def __init__(self, points: torch.Tensor, speeds: torch.Tensor = None, timesteps=0.1):
        """
        Args:
            points, timesteps: See `Trajectory`.
            speeds: optional, initial values for speeds along each axis, shape `(n-1,D)`
        """
        # TODO consider setting default timestep to 1
        super().__init__(points, timesteps)
        if speeds is not None:
            self.speeds = torch.nn.Parameter(speeds)
        else:
            # by default, initialise to speeds agreeing with other arguments
            with torch.no_grad():
                s = super().get_speed()
            self.speeds = torch.nn.Parameter(s)
        self._dynamics_matrix = None
        assert self.speeds.shape[1:] == self.points.shape[1:]
        assert self.speeds.shape[0] == self.points.shape[0]-1

    @classmethod
    def from_Trajectory(cls, trajectory: Trajectory):
        return PhaseSpaceTrajectory(points=trajectory.points, timesteps=trajectory._times)

    @classmethod
    def new_random(cls, num=50, d=2, dt=0.1, low=-1.0, high=1.0, rand_gen=None):
        """
        Returns:
            A new, completely random trajectory with `num` states/samples
            (whose velocities don't match their positions).
        """
        random_points = np.random.default_rng(rand_gen).uniform(low=low, high=high, size=(2, num, d))
        random_points = torch.Tensor(random_points).float()#.requires_grad_(True)
        return PhaseSpaceTrajectory(
            points = random_points[0],
            speeds = random_points[1,1:,:],
            timesteps = dt
        )

    def compute_speed_from_positions(self, override_timesteps=None):
        return super().get_speed(override_timesteps)

    def get_speed(self, override_timesteps=None, prepend=False):
        # NOTE Is this more or less confusing than leaving the
        # `super().get_speed()` method unchanged?
        # TODO override_timesteps does nothing, compatibility issue with Trajectory.get_speed
        s = self.speeds
        if prepend:
            z = torch.tensor([[0.0]*self.points.shape[1]]).to(DEVICE)
            s = torch.cat([z, s])
        return s

    def _get_coupling_matrix(self, fix_endpoints: bool = False):
        """
        Make a position-velocity coupling matrix E for a column vector state
        $Z = (X_0, ..., X_{n}, V_0, ..., V_{n-1})$, where
        each $X_i$ is a d-dimensional tuple of coordinates, and
        each $V_i$ is a d-dimensional tuple of velocity components
        along those coordinates.

        Args:
            fix_endpoints: if `True`, two constraints are added for initial and
                final positions; you should then add corresponding rows
                to other quantities in the loss (i.e. D in E@Z + D).
        """
        # n is no. of timesteps, d is no. of spatial coordinates
        n, d = self.points.shape
        A = torch.eye(d)
        # a block of d coordinate-velocity couplings all have the same timestep
        B = lambda i: torch.eye(d) * self.timesteps[i]
        # Dims of E:
        #  d*(n-1) rows, one for each constraint vi*dt = x(i+1)-xi, and maybe 2*d for two endpoints
        #  d*(2n-1) elements per row running over n*d coordinates and (n-1)*d speeds
        E = torch.zeros([d*(n-1+(2*fix_endpoints)), (2*n-1)*d]).to(DEVICE)
        for i in range(n-1):                      # there are n-1 constraints
            E[d*i:d*(i+1),d* i   :d*(i+1)] = A    # single out the position at i (diagonal)
            E[d*i:d*(i+1),d*(i+1):d*(i+2)] = -A   # subtract the position at i+1 (off-diagonal)
            j = d*(i+n)                           # n*d is the total size of the coordinate-part of the state
            E[d*i:d*(i+1),j:j+d] = B(i)           # add Δt * (speed at i)
        # ... so we end up with x_i - x_(i+1) + v_i * dt,
        # which we will set equal to 0 to get v_i = (x_(i+1) - x_i)/dt
        if fix_endpoints:
            # the final two constraints are for initial and final position
            E[d*(n-1):d*n, 0:d] = A             # initial coord block at index 0:d
            E[d*n:d*(n+1), d*(n-1):d*n] = A     # final coord block at index (n-1)d:nd
        return E

    def get_dynamic_loss(self, fix_endpoints=None):
        """
        Args:
            fix_endpoints: Optional tuple of points, which the trajectory is to
                start and end at, respectively. If omitted, the trajectory's
                start- and end-point remain unconstrained (by this loss).
                You should decide about this value only once.
        """
        # avoid re-creating the dynamics matrix every time
        if self._dynamics_matrix is None:
            if fix_endpoints is None:
                self._dynamics_matrix = self._get_coupling_matrix(False)
            else:
                self._dynamics_matrix = self._get_coupling_matrix(True)
        E = self._dynamics_matrix
        n, d = self.points.shape

        # Assemble all state variables in a 1D vector. Using torch operations
        # means the gradient of `points` and `speeds` is tracked.
        Z = torch.concat([torch.flatten(self.points), torch.flatten(self.speeds)], dim=0)

        # DEBUG check dimensions of E via assertions - remove?
        if fix_endpoints is not None:
            assert E.shape[0] == d*(n+1), "Inconsistency: can only decide once whether to fix endpoints in the matrix loss."
            x0, xf = fix_endpoints
            def tensorify(obj):
                if isinstance(obj, torch.Tensor):
                    return obj
                return torch.Tensor(obj).to(DEVICE)
            x0 = tensorify(x0)
            xf = tensorify(xf)
            # prepare the constant term in the loss E@Z - C
            # only important for endpoints, really
            C = torch.zeros(E.shape[0]).to(DEVICE)
            C[-2*d:-d] = x0
            C[-d:] = xf
            return (torch.matmul(E, Z) - C).pow(2).sum()
        else:
            assert E.shape[0] == d*(n-1), "Inconsistency: can only decide once whether to fix endpoints in the matrix loss."
            return (torch.matmul(E, Z)).pow(2).sum()

    @classmethod
    def new_line(cls, start, stop, num, dt, rand=None, rand_gen=None, come_back=False):
        xys = Trajectory.new_line(start, stop, num, dt, rand, rand_gen, come_back)
        return cls.from_Trajectory(xys)

    def detach_clone(self):
        if self._dt_cst:
            t = self._times
        else:
            t = self._times.clone()
        return PhaseSpaceTrajectory(self.points.detach().clone(), self.speeds.detach().clone(), t)

    def _reach_points_with_speed(self, points):
        """
        Transform for constraints, containing distances to given points
        (indices 0 to len(points)-1) and total speed (index len(points)) at each
        trajectory sample. Speed at the starting point is zero.

        Returns:
            tensor of shape (n, len(points))
        """
        pts = torch.stack(list(map(lambda p: p.p, points)), dim=0).to(self.points.dtype)
        distances = torch.cdist(self.points, pts)
        # prepend a single zero (for the meaningless speed of the first point)
        total_speeds = L2(self.get_speed(prepend=True)).unsqueeze(-1)
        # total_speeds = torch.concatenate([torch.zeros(1), L2(self.speeds)], dim=0).unsqueeze(-1)
        # concatenate speeds in the state dimension, then unsqueeze for batch dimension
        return torch.concatenate([distances, total_speeds], dim=-1).unsqueeze(0)

    # Highly specific to our setup - 4 points
    @classmethod
    def double_loop_conjoined(cls, speed_limit, points, reach_margin=0.0, smoothing_parameter=0.01):
        """
        Construct a constraint that reaches 4 `points` in swapped pairs
        (to make loops). I.e. if passing `[p1,p2,p3,p4]`, the trajectory will
        reach p2->p1, and separately p4->p3. Learned speed parameters are encouraged to
        stay below `speed_limit`.
        """
        # get the desired state dimension: a distance per point, plus total speed, plus constants
        # state = (*distances, speed, speed_limit, reach_margin)
        # better/general approach in `reach_sequence` below
        assert len(points) == 4, "must rewrite LTL!"
        const_ind = len(points) + 1     # this is the index of the first constant (first element after distances and speed)
        return Constraint(
            ltl = f"F (1 <= {const_ind} & X (F 0 <= {const_ind})) & F (3 <= {const_ind} & X (F 2 <= {const_ind})) & G({len(points)} <= {const_ind+1})",
            transform = lambda x: x._reach_points_with_speed(points),
            consts = [reach_margin, speed_limit],
            exectype = "ocaml",
            gamma = smoothing_parameter
        )

    #TODO combine with method above?
    @classmethod
    def double_loop_nested(cls, speed_limit, points, reach_margin=0.0, smoothing_parameter=0.01):
        """
        Construct a constraint that reaches 4 `points` in pairwise swapped order
        (to make loops). I.e. if passing `[p1,p2,p3,p4]`, the trajectory will
        reach p2,p1,p4,p3 in order. Learned speed parameters are encouraged to
        stay below `speed_limit`.
        """
        # get the desired state dimension: a distance per point, plus total speed, plus constants
        # state = (*distances, speed, speed_limit, reach_margin)
        assert len(points) == 4, "must rewrite LTL!"
        const_ind = len(points) + 1     # this is the index of the first constant (first element after distances and speed)
        return Constraint(
            ltl = f"F (1 <= {const_ind} & X (F (0 <= {const_ind} & X (F (3 <= {const_ind} & X (F (2 <= {const_ind}))))))) & G({len(points)}<={const_ind+1})",
            transform = lambda x: x._reach_points_with_speed(points),
            consts = [reach_margin, speed_limit],
            exectype = "ocaml",
            gamma = smoothing_parameter
        )

    # @classmethod
    # def reach_with_speed_limit(cls, points, speed_limit, reach_margin=None, smoothing_parameter=0.01):
    #     """
    #     Construct a constraint that reaches some `points` in order.
    #     I.e. if passing `points=[p1,p2,p3,p4]`, the trajectory will
    #     reach p1,p2,p3,p4 in order. *Learned* speed parameters are encouraged to
    #     stay below `speed_limit`.
    #     """
    #     if reach_margin is None:
    #         reach_margin = [0.0]*len(points)
    #     # state = (*distances, speed, speed_limit, reach_margin)
    #     l = len(points)     # number of distances per sample/state
    #     o = 1               # one total speed per sample/state
    #     ltl = ltl_reach_sequence(i=0, l=l, o=o)
    #     const_ind = l + o   # this is the index of the first constant (first element after distances and speed)
    #     return Constraint(
    #         ltl = f"({ltl}) & G({l} <= {l+o+1})",   # l+o+1 is the speed limit: l distances, o total speeds, 
    #         transform = lambda x: x._reach_points_with_speed(points=points),
    #         consts = [reach_margin, speed_limit],
    #         exectype = "ocaml",
    #         gamma = smoothing_parameter
    #     )



class FixedEndpointTrajectory(Trajectory):
    """
    Trajectory with fixed, gradient-disabled endpoints.
    """
    def __init__(self, z0, Z, zf, dt):
        super().__init__(Z, dt)
        self.z0 = z0
        self.zf = zf
        self.z0.requires_grad_(False)
        self.zf.requires_grad_(False)

    @classmethod
    def from_Trajectory(cls, trajectory):
        """
        Not a deep copy. Gradients of points that are not endpoints are tracked
        across this operation!
        """
        # assert trajectory._dt_cst
        p = trajectory.points.detach().clone()
        return cls(
            z0 = p[0].unsqueeze(0),
            Z = trajectory.points[1:-1],
            zf = p[-1].unsqueeze(0),
            dt = trajectory._times
        )

    # TODO not sure what to do with gradients here
    # def to_Trajectory(self, gradient=True):
    #     return Trajectory(
    #         points = torch.cat((self.z0, self.Z, self.zf), dim=0),
    #         timesteps = self._times
    #     )

    @property
    def Z(self):
        # for back-compatibility
        return self.points

    @classmethod
    def new_line(cls, start, stop, num, dt, come_back=False, rand=None, rand_gen=None):
        return cls.from_Trajectory(Trajectory.new_line(start, stop, num, dt, rand=rand, rand_gen=rand_gen, come_back=come_back))

    def detach_clone(self):
        if self._dt_cst:
            t = self._times
        else:
            t = self._times.clone()
        return FixedEndpointTrajectory(self.z0.clone(), self.Z.detach().clone(), self.zf.clone(), t)

    def forward(self):
        return torch.cat([self.z0, self.Z, self.zf], dim=0)

    ### the following methods are mostly for back-compatibility ###

    def zero_row_vec(self):
        return torch.tensor([[0.0]*self.Z.shape[1]])

    @staticmethod
    def _attach_endpoints(Z, x0, xf):
        return torch.cat((x0, Z, xf), dim=0)

    def get_position(self, in_constraint=False):
        x = self._attach_endpoints(self.Z, self.z0, self.zf)
        if in_constraint:
            return x.unsqueeze(0)
        return x

    @property
    def shape(self):
        return self.get_position().shape

    def get_displacements(self, in_constraint=False):
        x = self.get_position()
        d = torch.cat([self.zero_row_vec(), x[1:] - x[:-1]])
        if in_constraint:
            return d.unsqueeze(0)
        return d

    def get_speed(self, in_constraint=False):
        """
        Obtain a list of velocities.

        Prepend 0 speed - this means speed v_i = (x_i - x_(i-1)) / dt
        as opposed to (x_(i+1) - x_i) / dt
        """
        x = self.get_position()
        s = (x[1:] - x[:-1]) / self._times
        s = torch.cat([self.zero_row_vec(), s])
        if in_constraint:
            return s.unsqueeze(0)
        return s

    def get_acceleration(self, in_constraint=False):
        s = self.get_speed()
        a = (s[1:] - s[:-1]) / self._times
        a = torch.cat([self.zero_row_vec(), a])
        if in_constraint:
            return a.unsqueeze(0)
        return a

    @staticmethod
    ### return dimension is Z.shape[:-1], i.e. the spatial dimension is summed over
    # TODO replace with eucl_dist or L2 util method
    def _sqrt_sum_sqr(Z):
        return torch.sqrt(torch.sum(Z**2, dim=-1))

    def total_displacement(self, in_constraint=False):
        lengths = self._sqrt_sum_sqr(self.get_displacements())
        if in_constraint:
            return lengths.unsqueeze(-1).unsqueeze(0)
        return lengths

    def total_speed(self, in_constraint=False):
        speed = self._sqrt_sum_sqr(self.get_speed())
        if in_constraint:
            return speed.unsqueeze(-1).unsqueeze(0)
        return speed

    def total_acceleration(self, in_constraint=False):
        acceleration = self._sqrt_sum_sqr(self.get_acceleration())
        if in_constraint:
            return acceleration.unsqueeze(-1).unsqueeze(0)
        return acceleration

    ### returns a scalar
    def integral_length(self):
        lengths = self.total_displacement()
        # if in_constraint:
        #     return torch.sum(lengths).unsqueeze(0)
        return torch.sum(lengths)

    def distance_from(self, p, in_constraint=False):
        d = L2(self.get_position() - p)
        if in_constraint:
            return d.unsqueeze(-1).unsqueeze(0)
        return d        



### Obstacles ###
#################

# def dist_f(p):
#     d = lambda x : torch.sqrt(torch.sum(
#         (get_position(x) - p)**2,
#         dim=-1
#     )).unsqueeze(0).unsqueeze(-1) # Unsqueeze twice: once for batching, and once for state dimension (which just got summed over).
#     return d

class RectangleXY:
    def __init__(self, x_min=0, x_max=0, y_min=0, y_max=0):
        assert (x_max >= x_min), "wrong x order"
        assert (y_max >= y_min), "wrong y order"
        half_diff = torch.Tensor([(x_max-x_min)/2, (y_max-y_min)/2]).to(DEVICE)
        self.radius = half_diff
        self.centre = torch.Tensor([x_min, y_min]).to(DEVICE) + half_diff

    def scale(self, s):
        self.centre = self.centre*s
        self.radius = self.radius*s

    # def from_corner(xy, width, height):
    #     return Self()

    # def from_centre(centre, x, y):

    def avoid(self, margins=(0,0), gamma=0.01, exec_type="ocaml"):
        m = torch.Tensor(margins).to(DEVICE) + self.radius
        return Constraint(
            "G ({x_deviation} >= {x_margin} | {y_deviation} >= {y_margin})".format(
                x_deviation=0, x_margin=2,
                y_deviation=1, y_margin=3
            ),
            lambda Z: torch.abs(Z.get_position() - self.centre),
            [m[0],m[1]],  # TODO better way?
            gamma,
            exec_type
        )

    def to_patch(self, **kwargs):
        return plt.Rectangle(
            xy = self.centre-self.radius,
            width = 2*self.radius[0],
            height = 2*self.radius[1],
            **kwargs
        )

    def plot(self, axis, label, edgecolor='r', fill=False, linestyle="dotted", **kwargs):
        axis.add_artist(
            self.to_patch(edgecolor=edgecolor, fill=fill, linestyle=linestyle, **kwargs)
        )
        axis.text(self.centre[0], self.centre[1], label, ha="center", **kwargs)

class Point:
    def __init__(self, x=0, y=0):
        self.p = torch.Tensor([x,y]).to(DEVICE)

    def scale(self, s):
        self.p = self.p*s

    @classmethod
    def from_tensor(cls, pt):
        return Point(pt[0], pt[1])

    def reach_and_stay(self, margin=0, gamma=0.005, exec_type="ocaml"):
        return Constraint(
            "F (G (0 <= 1))",
            lambda x: x.distance_from(self.p, in_constraint=True),
            [margin],
            gamma,
            exec_type
        )

    def plot(self, axis, label, color="g", **kwargs):
        axis.scatter([self.p[0]], [self.p[1]], label=label, color=color, **kwargs)

def point_distances(Z, points):
    """
    Compute distance between each trajectory sample and each point in a given list.

    Args:
        Z: a `FixedEndpointTrajectory` with `n` samples (including endpoints)
        points: an ordered `list` of `Point`s, of length `p`

    Returns:
        tensor of shape `(n, p)`
    """
    pts = list(map(lambda p: p.p, points))
    pts = torch.stack(pts, dim=0).to(Z.Z.dtype)
    return torch.cdist(Z.get_position(), pts)

def ltl_reach_sequence(i, l, o=0):
    """
    Recursive construction of constraint string/formula.
    Index `j` will be the distance from the `j`-th point,
    while margins should be appended to each state sample (e.g. `j+len(points)`).

    Args:
        i: starting index
        l: number of points to reach (strict ending index)
        o: offset between final distance variable and first margin constant
            (e.g. if there are state variables like speed that aren't considered here)

    Returns:
        LTL constraint as a string for use with `ltl.constraint.Constraint`
    """
    if i==l-1:
        if stay:
            return "F (G ({distance} <= {margin}))".format(
                distance=i,
                margin=i+l+o
            )
        else:
            return "F ({distance} <= {margin})".format(
                distance=i,
                margin=i+l+o
            )
    return "F ({distance} <= {margin} & X ({nested_constraint}))".format(
            distance=i,
            margin=i+l+o,
            nested_constraint=ltl_reach_sequence(i+1)
        )

def reach_sequence(points, margins=None, stay=False, gamma=0.05, exec_type="ocaml"):
    """
    Args:
        points: an ordered list of points, to be reached in sequence
        margins: an ordered list of floats, specifying how closely each point should be reached
        stay: bool, True constrains the trajectory to eventually stay at the last point after reaching it.
            (Exercise: Does this imply not moving away between reaching the last point and ending the trajectory?)
    """
    if margins is None:
        margins = [0.0]*len(points)
    ltl_constraint = ltl_reach_sequence(0, len(points))
    
    return Constraint(
        ltl_constraint,
        lambda Z: point_distances(Z, points).unsqueeze(0),
        margins,
        gamma,
        exec_type
    )
