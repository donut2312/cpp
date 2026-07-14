import numpy as np
import matplotlib.pyplot as plt
import polarity_solver as ps
from polarity_solver import (pts, area, nb_i, nb_j, lap, plot_sphere,
                             N, beta, alpha, delta2)

#  paper uses a different threshold
ps.gamma = 0.35
hill = ps.hill

eps = 1.0
tau_c = 2.0

theta = np.arccos(np.clip(pts[:, 2], -1, 1))
phi = np.arctan2(pts[:, 1], pts[:, 0])

# unit vector in the theta direction at each generator
e_th = np.stack([np.cos(theta)*np.cos(phi),
                 np.cos(theta)*np.sin(phi),
                 -np.sin(theta)], axis=1)
vel = (eps*np.sin(theta))[:, None]*e_th

# face normals: direction from cell i to cell j, projected onto the surface
dij = pts[nb_j] - pts[nb_i]
mid = 0.5*(pts[nb_i] + pts[nb_j])
mid /= np.linalg.norm(mid, axis=1)[:, None]
dij = dij - (np.sum(dij*mid, axis=1)[:, None])*mid
Lface = np.linalg.norm(dij, axis=1)
nrm = dij/Lface[:, None]

vface = 0.5*(vel[nb_i] + vel[nb_j])
vn = np.sum(vface*nrm, axis=1)


def div_flow(u):
    uu = np.where(vn > 0, u[nb_i], u[nb_j])
    f = vn*uu*Lface
    out = np.zeros_like(u)
    np.add.at(out, nb_i, f)
    np.add.at(out, nb_j, -f)
    return out/area


def rhs_flow(u, t):
    ubar = np.sum(u*area)/np.sum(area)
    C = 1 - 2*alpha*ubar
    r = delta2*lap(u) - u + C*(beta + hill(u))
    if t < tau_c:
        r = r - div_flow(u)
    return r


def run(u, t0, T, dt):
    t = t0
    n = int(round((T - t0)/dt))
    for s in range(n):
        k1 = rhs_flow(u, t)
        k2 = rhs_flow(u + 0.5*dt*k1, t + 0.5*dt)
        k3 = rhs_flow(u + 0.5*dt*k2, t + 0.5*dt)
        k4 = rhs_flow(u + dt*k3, t + dt)
        u = u + dt*(k1 + 2*k2 + 2*k3 + k4)/6.0
        t = t + dt
    return u, t


if __name__ == "__main__":
    # start from the uniform off state u_ = beta/(1 + 2*alpha*beta)
    u = np.full(N, beta/(1 + 2*alpha*beta))

    dt = 0.001
    t = 0.0
    for tau in [0, 2, 10, 50]:
        if tau > t:
            u, t = run(u, t, tau, dt)
        plot_sphere(u, "tau = %g" % tau, "conv_%g.png" % tau)
        print(tau, u.min(), u.max())
