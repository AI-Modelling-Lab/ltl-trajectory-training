import numpy as np
import torch

class CanonicalSystem():

    def __init__(self, dt, start_x=1.0, ax=5.0):
        # self.ax = -np.log(cutoff)
        self.ax = ax
        self.dt = dt
        self.start_x = start_x

def canonical_step(x, ax, dt, tau=1.0):
    return x - (ax * x * dt) * tau

def canonical_rollout(start_val, ax, dt, time_span=1.0, tau=1.0):
    T = int(time_span / dt)
    x_t = np.zeros(T)
    x_t[0] = start_val

    for t in range(1, T):
        x_t[t] = canonical_step(x_t[t-1], ax, dt, tau)

    return x_t

def rbf(x, h, c):
    return np.exp(-h * (x[:,None] - c)**2)

def rbf_torch(x, h, c):
    return torch.exp(-h * (x - c)**2)

class DMP():
    def __init__(self, num_basis_funcs, dt, d, device):
        self.device = device

        self.ay = 25
        self.by = self.ay / 4.0

        self.dt = dt
        self.n_basis_funcs = num_basis_funcs
        self.dims = d

        self.cs = CanonicalSystem(dt=self.dt)

        self.c = np.exp(-self.cs.ax * np.linspace(0, 1, self.n_basis_funcs))
        self.h = np.square(np.diff(self.c) * 0.55)
        self.h = 1.0 / (np.append(self.h, self.h[-1]))

        # Torch equivalents
        self.t_h = torch.from_numpy(self.h).to(dtype=torch.float, device=torch.device(self.device))
        self.t_c = torch.from_numpy(self.c).to(dtype=torch.float, device=torch.device(self.device))


    def step(self, x, y, dy, tau=1.0):
        # step canonical system
        x_next = canonical_step(x, self.cs.ax, self.cs.dt, tau)

        # generate basis function activation
        psi = rbf(np.array([x_next]), self.h, self.c)

        f = np.zeros(self.dims)
        for d in range(self.dims):
            # generate the forcing term
            f[d] = x_next * (np.dot(psi, self.weights[d])) / np.sum(psi)

        # DMP acceleration
        # Sugar notation to agree with Pastor (2008) notation
        K = self.ay * self.by
        D = self.ay

        # Modified generalized
        ddy_next = (K * (self.goal - y) - D * dy / tau - K * (self.goal - self.y0) * x_next + K * f) * tau

        # Original Form
        # ddy_next = K * (self.goal - y) - D * dy + (self.goal - self.y0) * f

        # DMP Velocity
        dy_next = dy + (ddy_next * self.dt) * tau
        y_next = y + dy_next * self.dt

        return x_next, y_next, dy_next, ddy_next

    def step_torch(self, starts, goals, x, y, dy, weights, tau=1.0):
        # step canonical system
        x_next = canonical_step(x, self.cs.ax, self.cs.dt, tau)

        # generate basis function activation
        psi = rbf_torch(x_next, self.t_h, self.t_c)

        f = x_next * torch.matmul(weights, psi) / torch.sum(psi)

        # DMP acceleration
        # Sugar notation to agree with Pastor (2008) notation
        K = self.ay * self.by
        D = self.ay

        # Modified generalized
        ddy_next = (K * (goals - y) - D * dy / tau - K * (goals - starts) * x_next + K * f) * tau

        # Original Form
        # ddy_next = K * (self.goal - y) - D * dy + (self.goal - self.y0) * f

        # DMP Velocity
        dy_next = dy + (ddy_next * self.dt) * tau
        y_next = y + dy_next * self.dt

        return x_next, y_next, dy_next, ddy_next


    def rollout(self, time_span=1.0, tau=1.0):
        scaled_tsteps = int((time_span / self.dt) / tau)
        # set up tracking vectors
        y_track = np.zeros((scaled_tsteps, self.dims))
        dy_track = np.zeros((scaled_tsteps, self.dims))
        ddy_track = np.zeros((scaled_tsteps, self.dims))

        y_track[0] = self.y0

        x = self.cs.start_x

        for t in range(1, scaled_tsteps):
            x, y_track[t], dy_track[t], ddy_track[t] = self.step(x, y_track[t-1], dy_track[t-1], tau)

        return y_track, dy_track, ddy_track

    def rollout_torch(self, starts, goals, weights, time_span=1.0, tau=1.0):
        scaled_tsteps = int((time_span / self.dt) / tau)
        batch_size = starts.shape[0]
        y_track = torch.zeros((batch_size, scaled_tsteps, self.dims), device=torch.device(self.device))
        dy_track = torch.zeros((batch_size, scaled_tsteps, self.dims), device=torch.device(self.device))
        ddy_track = torch.zeros((batch_size, scaled_tsteps, self.dims), device=torch.device(self.device))

        y_track[:, 0] = starts.reshape(batch_size, -1)

        x = self.cs.start_x

        for t in range(1, scaled_tsteps):
            x, y_track[:, t], dy_track[:, t], ddy_track[:,t] = self.step_torch(
                starts.view(batch_size, -1),
                goals.view(batch_size, -1),
                x,
                y_track[:, t-1],
                dy_track[:, t-1],
                weights,
                tau)

        return y_track, dy_track, ddy_track
