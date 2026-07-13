import numpy as np
from scipy.spatial import SphericalVoronoi
import matplotlib.pyplot as plt

alpha = 1.0
beta  = 0.1
gamma = 0.3
delta2 = 0.05
nu = 20.0

N = 2000
R = 1.0


def fib_points(n):
    i = np.arange(n)
    z = 1 - 2*(i + 0.5)/n
    r = np.sqrt(1 - z*z)
    phi = (i + 0.5) * np.pi * (3 - np.sqrt(5))
    x = r*np.cos(phi)
    y = r*np.sin(phi)
    return np.stack([x, y, z], axis=1)


pts = fib_points(N)
sv = SphericalVoronoi(pts, R, np.zeros(3))
sv.sort_vertices_of_regions()
area = sv.calculate_areas()

# build neighbour list: two generators are neighbours if their regions share
# two vertices (an edge)
edge_map = {}
for i, reg in enumerate(sv.regions):
    m = len(reg)
    for k in range(m):
        a = reg[k]
        b = reg[(k+1) % m]
        key = (a, b) if a < b else (b, a)
        edge_map.setdefault(key, []).append(i)

nb_i = []
nb_j = []
w = []
for key, cells in edge_map.items():
    if len(cells) != 2:
        continue
    i, j = cells
    v1 = sv.vertices[key[0]]
    v2 = sv.vertices[key[1]]
    # length of the shared edge (great circle arc)
    c = np.clip(np.dot(v1, v2)/(R*R), -1, 1)
    L = R*np.arccos(c)
    # distance between the two generators
    c2 = np.clip(np.dot(pts[i], pts[j])/(R*R), -1, 1)
    d = R*np.arccos(c2)
    nb_i.append(i); nb_j.append(j); w.append(L/d)

nb_i = np.array(nb_i)
nb_j = np.array(nb_j)
w = np.array(w)


def lap(u):
    f = w*(u[nb_j] - u[nb_i])
    out = np.zeros_like(u)
    np.add.at(out, nb_i, f)
    np.add.at(out, nb_j, -f)
    return out/area


def hill(u):
    up = np.abs(u)**nu
    return up/(up + gamma**nu)


def rhs(u):
    ubar = np.sum(u*area)/np.sum(area)
    C = 1 - 2*alpha*ubar
    return delta2*lap(u) - u + C*(beta + hill(u))


def run(u, T, dt):
    nsteps = int(T/dt)
    for s in range(nsteps):
        k1 = rhs(u)
        k2 = rhs(u + 0.5*dt*k1)
        k3 = rhs(u + 0.5*dt*k2)
        k4 = rhs(u + dt*k3)
        u = u + dt*(k1 + 2*k2 + 2*k3 + k4)/6.0
    return u


def plot_sphere(u, title, fname):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    for i, reg in enumerate(sv.regions):
        poly = sv.vertices[reg]
        col = plt.cm.jet((u[i] - u.min())/(u.max() - u.min() + 1e-12))
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        pc = Poly3DCollection([poly], color=col)
        ax.add_collection3d(pc)
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
    ax.set_axis_off()
    ax.set_title(title)
    ax.view_init(20, 30)
    plt.savefig(fname, dpi=110, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    rng = np.random.default_rng(3)
    # smooth random field: sum of a few low order modes, then rescale so the
    # mean stays below 1/(2*alpha)
    u = np.zeros(N)
    for k in range(6):
        d = rng.normal(size=3)
        d /= np.linalg.norm(d)
        u += rng.normal()*np.exp(2.0*(pts @ d))
    u -= u.min()
    u *= 0.45/u.max()

    dt = 0.002
    snaps = [0, 0.5, 1, 2, 5, 10, 15, 20, 50]
    t = 0.0
    for tau in snaps:
        if tau > t:
            u = run(u, tau - t, dt)
            t = tau
        plot_sphere(u, "tau = %g" % tau, "snap_%g.png" % tau)
        print(tau, u.min(), u.max(), np.sum(u*area)/np.sum(area))
