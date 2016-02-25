from casadi import *
import time
from collections import OrderedDict
from helpers import casadi_vec, casadi_struct, casadi_vec2struct, casadi_struct2vec
from helpers import skew

# Environment
g = 9.81     # [N/kg]

# platform charateristics
ma = 0.5          # Mass of platform [kg]
mb = 0.1          # Mass of end-effector [kg] 
L = 0.25          # Reference length [m]
L_pendulum = 0.20 # [m]

I_max = 0.1 * L**2 #Inertia of a point mass at a distance L
I_ref = I_max/5   

I = diag([I_ref/2,I_ref/2,I_ref]) # [N.m^2]

# propellor position
rotors_pos = blockcat([[L, 0, 0],[0, L, 0],[ -L, 0, 0],[0, -L, 0]]).T # rotor positions
rotors_N = 4

# System states

pa  = SX.sym('pa',3)  # platform position
dpa = SX.sym('dpa',3) # platform velocity wrt inertial

R = SX.sym('R',3,3)   # Rotation matrix from body to inertial
w = SX.sym('w',3)     # Angular rate, expressed in body
dw = SX.sym('dw',3)   # Angular acceleration, expressed in body

# System controls
r = SX.sym('r',rotors_N) # rotor action [N]

# Lagrange mechanics for the translation part
q   = pa
dq  = dpa
ddq = SX.sym('ddq',*q.shape)

# Potential energy
E_pot = ma*g*pa[2]

# Kinetic energy
E_kin = 0.5*ma*dot(dpa,dpa)

# Lagrange function for translation part
Lag = E_kin - E_pot

# Force-less Lagrange equations for translation part
eq = jtimes(gradient(Lag,dq),vertcat([q,dq]),vertcat([dq,ddq])) - gradient(Lag,q)

F_platform = 0
T_platform = 0

# Aerodynamics
for i in range(rotors_N):
    F_aero_body = r[i]*vertcat([0,0,1])
    F_platform = F_platform + mtimes(R,F_aero_body)
    T_platform = T_platform + cross(rotors_pos[:,i],F_aero_body)

# Residual for translation (from Lagrange)
res_transl = eq - F_platform
# Residual for rotation (Euler equations)
res_rot = mtimes(I,dw)+cross(w,mtimes(I,w))-T_platform

dae_x = OrderedDict([('q',q),('dq',dq),('R',R),('w',w)])
dae_z = OrderedDict([('ddq',ddq),('dw',dw)])
dae = {}
dae["x"] = casadi_struct2vec(dae_x)
dae["z"] = casadi_struct2vec(dae_z)
dae["p"] = r
dae["ode"] = casadi_vec(dae_x,q=dq,dq=ddq,R=mtimes(R,skew(w)),w=dw)
dae["alg"] = vertcat([res_transl,res_rot])

nx = dae["x"].shape[0]
nu = dae["p"].shape[0]
nz = dae["z"].shape[0]

# Nominal force per rotor needed to hold quadcopter stationary
r_nom = ma*g/rotors_N

x0_guess = casadi_vec(dae_x,q=vertcat([0,0,0]),R=DM.eye(3))
u_guess  = vertcat([r_nom]*rotors_N)
z_guess  = casadi_vec(dae_z,0)

T = 1.0
N = 14 # Caution; mumps linear solver fails when too large

options = {"implicit_solver": "newton","number_of_finite_elements":1,"interpolation_order":4,"collocation_scheme":"radau","implicit_solver_options": {"abstol":1e-9},"tf":T/N}

intg = integrator("intg","collocation",dae,options)
daefun = Function("daefun",dae,["x","z","p"],["ode","alg"])
print daefun([x0_guess,z_guess,u_guess])

Xs  = [MX.sym("X",nx) for i in range(N+1)]
Us  = [MX.sym("U",nu) for i in range(N)]

V_block = OrderedDict()
V_block["X"]  = Sparsity.dense(nx,1)
V_block["U"]  = Sparsity.dense(nu,1)

invariants = Function("invariants",[dae["x"]],[vertcat([triu(mtimes(R.T,R)-DM.eye(3)).nz[:]])])

# Simple bounds on states
lbx = []
ubx = []

# List of constraints
g = []

# List of all decision variables (determines ordering)
V = []
for k in range(N):
  # Add decision variables
  V += [casadi_vec(V_block,X=Xs[k],U=Us[k])]
  
  if k==0:
    # Bounds at t=0
    q_lb  = DM([0,0,0])
    q_ub  = DM([0,0,0])
    dq_lb = DM([0]*3)
    dq_ub = DM([0]*3)
    x_lb = casadi_vec(dae_x,-inf,q=q_lb,dq=dq_lb)
    x_ub = casadi_vec(dae_x,inf,q=q_ub,dq=dq_ub)
    lbx.append(casadi_vec(V_block,-inf,X=x_lb))
    ubx.append(casadi_vec(V_block,inf,X=x_ub))
  else:
    # Bounds for other t
    lbx.append(casadi_vec(V_block,-inf))
    ubx.append(casadi_vec(V_block,inf))
    
  # Obtain collocation expressions
  
  out = intg({"x0":Xs[k],"p":Us[k]})

  g.append(Xs[k+1]-out["xf"])

V+= [Xs[-1]]

# Bounds for final t
q_lb  = DM([1,0,0.5])
q_ub  = DM([1,0,0.5])
dq_lb = DM([0]*3)
dq_ub = DM([0]*3)
x_lb = casadi_vec(dae_x,-inf,q=q_lb,dq=dq_lb)
x_ub = casadi_vec(dae_x,inf,q=q_ub,dq=dq_ub)
lbx.append(x_lb)
ubx.append(x_ub)


# Construct regularisation
reg = 0
for x in Xs:
  xstruct = casadi_vec2struct(dae_x,x)
  reg += sumRows(sumCols((xstruct["R"]-DM.eye(3))**2))
  reg += sumRows(sumCols(xstruct["w"]**2))
  reg += sumRows(sumCols(xstruct["dq"]**2))
  
for u in Us:
  reg += 100*sumRows(sumCols((u-r_nom)**2))

e = vertcat(Us)
nlp = {"x":veccat(V), "f":dot(e,e)+reg, "g": vertcat([invariants([Xs[0]])[0]]+g)}

solver = nlpsol("solver","ipopt",nlp)

x0 = vertcat([x0_guess,u_guess]*N+[x0_guess])

args = {}
args["x0"] = x0
args["lbx"] = vertcat(lbx)
args["ubx"] = vertcat(ubx)
args["lbg"] = 0
args["ubg"] = 0

print vertcat(lbx)

res = solver(args)

res_split = vertsplit(res["x"],casadi_struct2vec(V_block).shape[0])

res_U = [casadi_vec2struct(V_block,r)["U"] for r in res_split[:-1]]

res_X = [casadi_vec2struct(V_block,r)["X"] for r in res_split[:-1]]+[res_split[-1]]

invariants_err = [invariants([r])[0] for r in res_X]

sol = [casadi_vec2struct(dae_x,r) for r in res_X]

from pylab import *

figure()
plot(horzcat([s["q"] for s in sol]).T)
figure()
plot(horzcat(res_U).T)

figure()
plot(horzcat(invariants_err).T)

show()

