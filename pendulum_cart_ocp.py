from casadi import *
import time
from collections import OrderedDict
from helpers import casadi_vec, casadi_struct, casadi_vec2struct, casadi_struct2vec
from helpers import skew

# Model properties
L = 0.58 # m
g = 9.81 # gravity
m = 80 # Trunk mass

# Variable declaration
p = casadi.SX.sym('p', 2) # CoP position
c = casadi.SX.sym('c', 3) # CoM position
dp = casadi.SX.sym('dp', 2) # CoP velocity
dc = casadi.SX.sym('dc', 3) # CoM velocity
u = casadi.SX.sym('u', 2) # Cart control forces
z = casadi.SX.sym('z') # Algebraic variable

# State variables
q = casadi.vertcat([p, c])
dq = casadi.vertcat([dp, dc])
ddq = SX.sym('ddq', *q.shape)
dz = casadi.SX.sym('dz')

# Lagrange mechanics
T = 0.5 * m * casadi.dot(dc, dc) # Kinetic energy
V = m * g * c[2] # Potential Energy
ext_p = casadi.vertcat([p, 0])
cp_diff = c - ext_p
C = casadi.dot(cp_diff, cp_diff) - L**2 # Algebraic constraint

# Lagrange function
Lag = T - V - z * C

# Force-less Lagrange equations
eq = jtimes(gradient(Lag,dq),vertcat([q,dq,z]),vertcat([dq,ddq,dz])) - gradient(Lag,q)

F_cart = casadi.vertcat([0, 0 , 0, u])
res_eq = eq - F_cart

# DAE definition
dae_x = OrderedDict([('q',q),('dq',dq)])
dae_z = OrderedDict([('z',z)])
dae = {}
dae["x"] = casadi_struct2vec(dae_x)
dae["z"] = casadi_struct2vec(dae_z)
dae["p"] = u
dae["ode"] = casadi_vec(dae_x,q=dq,dq=ddq)
dae["alg"] = vertcat([res_eq, C])

# DAE dimension
nx = dae["x"].shape[0]
nu = dae["p"].shape[0]
nz = dae["z"].shape[0]

# Nominal force per rotor needed to hold quadcopter stationary
u_nom = casadi.vertcat([0, 0])

x0_guess = casadi_vec(dae_x,q=casadi.vertcat([0,0,0,0,0]))
u_guess  = casadi.vertcat([0,0])
z_guess  = casadi_vec(dae_z, 0)

#print daefun([x0_guess,z_guess,u_guess])
#
#Xs  = [MX.sym("X",nx) for i in range(N+1)]
#Us  = [MX.sym("U",nu) for i in range(N)]
#
#V_block = OrderedDict()
#V_block["X"]  = Sparsity.dense(nx,1)
#V_block["U"]  = Sparsity.dense(nu,1)
#
#invariants = Function("invariants",[dae["x"]],[vertcat([triu(mtimes(R.T,R)-DM.eye(3)).nz[:]])])
#
## Simple bounds on states
#lbx = []
#ubx = []
#
## List of constraints
#g = []
#
## List of all decision variables (determines ordering)
#V = []
#for k in range(N):
#  # Add decision variables
#  V += [casadi_vec(V_block,X=Xs[k],U=Us[k])]
#  
#  if k==0:
#    # Bounds at t=0
#    q_lb  = DM([0,0,0])
#    q_ub  = DM([0,0,0])
#    dq_lb = DM([0]*3)
#    dq_ub = DM([0]*3)
#    x_lb = casadi_vec(dae_x,-inf,q=q_lb,dq=dq_lb)
#    x_ub = casadi_vec(dae_x,inf,q=q_ub,dq=dq_ub)
#    lbx.append(casadi_vec(V_block,-inf,X=x_lb))
#    ubx.append(casadi_vec(V_block,inf,X=x_ub))
#  else:
#    # Bounds for other t
#    lbx.append(casadi_vec(V_block,-inf))
#    ubx.append(casadi_vec(V_block,inf))
#    
#  # Obtain collocation expressions
#  
#  out = intg({"x0":Xs[k],"p":Us[k]})
#
#  g.append(Xs[k+1]-out["xf"])
#
#V+= [Xs[-1]]
#
## Bounds for final t
#q_lb  = DM([1,0,0.5])
#q_ub  = DM([1,0,0.5])
#dq_lb = DM([0]*3)
#dq_ub = DM([0]*3)
#x_lb = casadi_vec(dae_x,-inf,q=q_lb,dq=dq_lb)
#x_ub = casadi_vec(dae_x,inf,q=q_ub,dq=dq_ub)
#lbx.append(x_lb)
#ubx.append(x_ub)
#
#
## Construct regularisation
#reg = 0
#for x in Xs:
#  xstruct = casadi_vec2struct(dae_x,x)
#  reg += sumRows(sumCols((xstruct["R"]-DM.eye(3))**2))
#  reg += sumRows(sumCols(xstruct["w"]**2))
#  reg += sumRows(sumCols(xstruct["dq"]**2))
#  
#for u in Us:
#  reg += 100*sumRows(sumCols((u-r_nom)**2))
#
#e = vertcat(Us)
#nlp = {"x":veccat(V), "f":dot(e,e)+reg, "g": vertcat([invariants([Xs[0]])[0]]+g)}
#
#solver = nlpsol("solver","ipopt",nlp)
#
#x0 = vertcat([x0_guess,u_guess]*N+[x0_guess])
#
#args = {}
#args["x0"] = x0
#args["lbx"] = vertcat(lbx)
#args["ubx"] = vertcat(ubx)
#args["lbg"] = 0
#args["ubg"] = 0
#
#print vertcat(lbx)
#
#res = solver(args)
#
#res_split = vertsplit(res["x"],casadi_struct2vec(V_block).shape[0])
#
#res_U = [casadi_vec2struct(V_block,r)["U"] for r in res_split[:-1]]
#
#res_X = [casadi_vec2struct(V_block,r)["X"] for r in res_split[:-1]]+[res_split[-1]]
#
#invariants_err = [invariants([r])[0] for r in res_X]
#
#sol = [casadi_vec2struct(dae_x,r) for r in res_X]
#
#from pylab import *
#
#figure()
#plot(horzcat([s["q"] for s in sol]).T)
#figure()
#plot(horzcat(res_U).T)
#
#figure()
#plot(horzcat(invariants_err).T)
#
#show()
#
