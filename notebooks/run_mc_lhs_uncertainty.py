import numpy as np
from firedrake import UnitSquareMesh, FunctionSpace, TrialFunction, TestFunction
from firedrake import SpatialCoordinate, dx, pi, sin, dot, grad, DirichletBC, Constant
from firedrake import assemble, Function, solve

import stat_fem

from scipy.stats import multivariate_normal
from scipy.stats import norm as normal
from scipy.stats import gamma

from firedrake.petsc import PETSc
from pathlib import Path

from tqdm import tqdm

# fig_path = Path.home() / 'Dropbox' / 'phd' / 'projects' / 'stat-fem' / 'meeting-notes' / '2020-04-27-images'
fig_path = Path.home() / 'Dropbox' / 'phd' / 'projects' / 'stat-fem' / 'misc-notes' / 'lhs-uncertainty-images'


experiment_label = 'basic'


class ThetaPriorGamma:
    
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def sample(self):
        return gamma.rvs(self.a, scale=self.b)
    
    def logpdf(self, theta):
        return gamma.logpdf(theta, self.a, scale=self.b)
    
class LogThetaPriorNormal:
    
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

    def sample(self):
        return np.random.normal(self.loc, self.scale)
    
    def logpdf(self, log_theta):
        return normal.logpdf(log_theta, self.loc, self.scale)
    

def theta_proposal_logpdf(theta_from, theta_to, prop_std):
    return normal.logpdf(theta_to, theta_from, prop_std)

def theta_proposal_sample(theta, prop_std):
    return np.random.normal(theta, prop_std)


is_prior_logscale = False

if is_prior_logscale:
    theta_prior = LogThetaPriorNormal(0, 1)
else:
    theta_prior = ThetaPriorGamma(1, 1)


nx = 21

mesh = UnitSquareMesh(nx - 1, nx - 1)
V = FunctionSpace(mesh, "CG", 1)

M = mesh.coordinates.vector().dat.data.shape[0]

u = TrialFunction(V)
v = TestFunction(V)

f = Function(V)
x = SpatialCoordinate(mesh)
# f.interpolate(Constant(10))
f.interpolate(-(8*pi*pi)*sin(x[0]*pi*2)*sin(x[1]*pi*2))
# f.interpolate(10 + 5 * (sin(2 * pi * x[0]) + sin(2 * pi * x[1])))


L = f * v * dx
b = assemble(L)
b_np = b.vector().dat.data

bc = [DirichletBC(V, 1, 2), DirichletBC(V, 1, 4)]

log_sigma_f = np.log(1)
log_l_f = np.log(0.1)

G = stat_fem.ForcingCovariance(V, log_sigma_f, log_l_f, cutoff=1e-10)
G.assemble()
G_np = G.G.getValues(np.arange(G.nx).astype(np.int32), np.arange(G.nx).astype(np.int32))


mc_num_samples = 1_000_000
u_samples_mc = np.zeros((mc_num_samples, M))

u_trial = TrialFunction(V)
v = TestFunction(V)

for n in tqdm(range(mc_num_samples)):
    theta_n = theta_prior.sample()
    if is_prior_logscale:
        theta_n = np.exp(theta_n)

    a = (dot(grad(v), Constant(theta_n) * grad(u_trial))) * dx
    A = assemble(a, bcs=bc)
    u_mean = Function(V)
    solve(A, u_mean, b)
   
    u_samples_mc[n, :] = u_mean.vector().dat.data.copy()

if is_prior_logscale:
    np.save(f"MC_samples_{mc_num_samples}_lognormal_prior_nx_{nx}_{experiment_label}.npy", u_samples_mc)
else:
    np.save(f"MC_samples_{mc_num_samples}_gamma_prior_nx_{nx}_{experiment_label}.npy", u_samples_mc)
