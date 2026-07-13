import numpy as np
import matplotlib.pyplot as plt

beta = 0.1

NTERM = 400


def P(delta, x):
    # power series for the Legendre function P_mu(x), eq S9. only depends on
    # delta through 1/delta^2 so we never have to deal with complex mu
    s = np.ones_like(np.asarray(x, dtype=float))
    term = np.ones_like(s)
    for n in range(1, NTERM):
        term = term*(1 - x)*(1/delta**2 + (n-1) + (n-1)**2)/(2*n*n)
        s = s + term
        if np.max(np.abs(term)) < 1e-16*np.max(np.abs(s)):
            break
    return s


def dP(delta, x, h=1e-6):
    return (P(delta, x + h) - P(delta, x - h))/(2*h)


def ratio(delta, xc):
    return (dP(delta, xc)/dP(delta, -xc))*(P(delta, -xc)/P(delta, xc))


def gamma_of(xc, alpha, delta):
    r = ratio(delta, xc)
    num = 1 + beta*(1 + r)
    den = (1 + r)*(1 + alpha*(1 + xc) + 2*alpha*beta)
    return num/den


def k_of(xc, alpha, delta):
    r = ratio(delta, xc)
    g = gamma_of(xc, alpha, delta)
    return g*(1 + r)/(1 + beta*(1 + r))


def profile(xc, alpha, delta, x):
    g = gamma_of(xc, alpha, delta)
    k = k_of(xc, alpha, delta)
    u = np.zeros_like(x)
    hi = x > xc
    lo = ~hi
    u[hi] = (g - beta*k)*P(delta, x[hi])/P(delta, xc) + beta*k
    u[lo] = (g - (1+beta)*k)*P(delta, -x[lo])/P(delta, -xc) + (1+beta)*k
    return u


# --- panel A : three roots of eq 11 and the profiles they give
alpha = 1.0
delta = 0.3
gstar = 0.25

xc = np.linspace(-0.98, 0.98, 2000)
gg = gamma_of(xc, alpha, delta)

roots = []
for i in range(len(xc)-1):
    if (gg[i]-gstar)*(gg[i+1]-gstar) < 0:
        # linear interpolation is good enough on this grid
        t = (gstar - gg[i])/(gg[i+1] - gg[i])
        roots.append(xc[i] + t*(xc[i+1] - xc[i]))
print("roots:", roots)

fig, ax = plt.subplots(2, 1, figsize=(5, 7))
ax[0].plot(xc, gg, 'k')
ax[0].axhline(gstar, ls='--', color='k', lw=0.8)
ax[0].plot(roots, [gstar]*len(roots), 'ro')
ax[0].set_xlabel('eta_c')
ax[0].set_ylabel('gamma(eta_c)')

x = np.linspace(-0.999, 0.999, 600)
for rt in roots:
    ax[1].plot(x, profile(rt, alpha, delta, x))
ax[1].axhline(gstar, ls='--', color='k', lw=0.8)
ax[1].set_xlabel('eta')
ax[1].set_ylabel('u(eta)')
plt.tight_layout()
plt.savefig('fig3a.png', dpi=110)
plt.close()


# --- panel B : cusps in the (gamma, delta) plane for a few alpha
# stable caps sit on the decreasing branch of gamma(eta_c), so the edge of
# stability is where gamma(eta_c) turns around
t = np.linspace(-0.985, 0.985, 1200)


def turning_points(alpha, delta):
    g = gamma_of(t, alpha, delta)
    dg = np.diff(g)
    out = []
    for i in range(len(dg)-1):
        if dg[i]*dg[i+1] < 0:
            out.append(g[i+1])
    return out


plt.figure(figsize=(6, 4))
for alpha in [0.25, 0.5, 1.0, 2.0]:
    lo, hi, dl = [], [], []
    for d in np.linspace(0.03, 0.9, 120):
        e = turning_points(alpha, d)
        if len(e) == 2:
            dl.append(d)
            lo.append(min(e))
            hi.append(max(e))
    if not dl:
        continue
    plt.plot(lo, dl, 'k')
    plt.plot(hi, dl, 'k')
    plt.text(hi[0], dl[0]+0.01, 'a=%g' % alpha, fontsize=8)
plt.xlabel('gamma')
plt.ylabel('delta')
plt.xlim(0.1, 0.6)
plt.tight_layout()
plt.savefig('fig3b.png', dpi=110)
plt.close()
print("done")
