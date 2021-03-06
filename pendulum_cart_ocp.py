from casadi import *
import time
from collections import OrderedDict
from helpers import casadi_vec, casadi_struct, casadi_vec2struct, casadi_struct2vec
from helpers import skew

# Model properties
L = 0.58 # m
g = 9.81 # gravity
m_trunk = 80 # Trunk mass
m_cart = 2  # Virtual cart mass for matrix inversion

# Variable declaration
cop = casadi.SX.sym('cop', 2) # CoP position
com = casadi.SX.sym('com', 3) # CoM position
dcop = casadi.SX.sym('dcop', 2) # CoP velocity
dcom = casadi.SX.sym('dcom', 3) # CoM velocity
u = casadi.SX.sym('u', 2) # Cart control forces
z = casadi.SX.sym('z') # Algebraic variable

# State variables
q = casadi.vertcat([cop, com])
dq = casadi.vertcat([dcop, dcom])
dz = casadi.SX.sym('dz')

# Lagrange mechanics
T = 0.5 * m_trunk * casadi.dot(dcom, dcom) + 0.5 * m_cart * casadi.dot(dcop, dcop) # Kinetic energy
V = m_trunk * g * com[2] # Potential Energy
ext_cop = casadi.vertcat([cop, 0])
point_diff = com - ext_cop
c = 0.5 * (casadi.dot(point_diff, point_diff) - L**2) # Algebraic constraint

# Lagrange function
Lag = T - V - z * c
Fg_cart = casadi.vertcat([u, 0, 0, 0])

# DAE equation: [dot(q) == dq;
#                dot(dq) == ddq;
#                   0    == c]

# Computing the ODE
Ldq = casadi.jacobian(Lag, dq)
Lq = casadi.jacobian(Lag, q)
M = DM(casadi.jacobian(Ldq, dq)).full() # Inertial matrix
# Note that Mdot = casadi.jacobian(casadi.jacobian(Ldq, dq), q) = 0
gradC_z = casadi.jacobian(c, q) * z
ddq = casadi.mtimes(casadi.inv(M), Lq.T + Fg_cart - gradC_z.T) ## Ode response explicitly defined

# Index reduction
dc = casadi.mtimes(casadi.jacobian(c, q), dq)
ddc = casadi.mtimes(casadi.jacobian(dc, q), dq) + casadi.mtimes(casadi.jacobian(dc, dq), ddq)

print DM(casadi.jacobian(c, z))
print DM(casadi.jacobian(dc, z))
print DM(casadi.jacobian(ddc, z))

# DAE definition
dae_x = OrderedDict([('q',q), ('dq',dq)])
dae_z = OrderedDict([('z',z)])
dae = {}
dae["x"] = casadi_struct2vec(dae_x)
dae["z"] = casadi_struct2vec(dae_z)
dae["p"] = u
dae["ode"] = casadi_vec(dae_x, q=dq, dq=ddq)
dae["alg"] = ddc + dc + c

# DAE dimension
nx = dae["x"].shape[0]
nu = dae["p"].shape[0]
nz = dae["z"].shape[0]

# Nominal force per rotor needed to hold quadcopter stationary
u_nom = casadi.vertcat([0, 0])

x0_guess = casadi_vec(dae_x, q=casadi.vertcat([0, 0, 0, 0, L]))
u_guess  = casadi.vertcat([0, 0])
z_guess  = casadi_vec(dae_z, 0)

T = 1.0
N = 14 # Caution; mumps linear solver fails when too large

#options = {"implicit_solver": "newton",
#           "number_of_finite_elements": 1,
#           "interpolation_order": 4,
#           "collocation_scheme": "radau",
#           "implicit_solver_options": {"abstol":1e-9},
#           "tf": T/N}
options = {"abstol": 1e-9,
           "tf": T/N}

intg = casadi.integrator("intg", "idas", dae, options)
daefun = Function("daefun", dae, ["x","z","p"], ["ode","alg"])

rf = rootfinder('rf','newton', daefun, {"implicit_input":1,"implicit_output":1})
rfsol = rf({'x':x0_guess, 'z':z_guess, 'p':u_guess})
z_con = rfsol['alg']
print ' --- Consistent initial conditions ---'
print 'z_0 = ', z_con


Jdaefun = daefun.jacobian('z','alg')
print Jdaefun({'x': x0_guess, 'z': z_con, 'p': u_guess})
print daefun({'x': x0_guess, 'z': z_con, 'p': u_guess})

Xs = [MX.sym("X",nx) for i in range(N+1)]
Us = [MX.sym("U",nu) for i in range(N)]
#Zs = [MX.sym("Z",nz) for i in range(N+1)]

V_block = OrderedDict()
V_block["X"]  = Sparsity.dense(nx, 1)
V_block["U"]  = Sparsity.dense(nu, 1)
#V_block["Z"]  = Sparsity.dense(nz, 1)

invariants = Function("invariants",
                      [dae["x"]],
                      [vertcat([c,dc])])
                      #[casadi.mtimes((point_diff.T, point_diff))])

# Simple bounds on states
lbx = []
ubx = []

# List of constraints
g = []

# List of all decision variables (determines ordering)
V = []
for k in range(N):
    # Add decision variables
    V += [casadi_vec(V_block, X=Xs[k], U=Us[k])]#Z=Zs[k], U=Us[k])]
  
    if k == 0:
        # Bounds at t=0
        q_lb  = DM([0]*5)
        q_ub  = DM([0]*5)
        dq_lb = DM([0]*5)
        dq_ub = DM([0]*5)
        x_lb = casadi_vec(dae_x, -inf, q=q_lb, dq=dq_lb)
        x_ub = casadi_vec(dae_x, inf, q=q_ub, dq=dq_ub)
        lbx.append(casadi_vec(V_block, -inf, X=x_lb))
        ubx.append(casadi_vec(V_block, inf, X=x_ub))
    else:
        # Bounds for other t
        lbx.append(casadi_vec(V_block, -inf))
        ubx.append(casadi_vec(V_block, inf))
    
    # Obtain collocation expressions
    out = intg({"x0": Xs[k], "z0": z_con, "p": Us[k]})

    g.append(Xs[k+1] - out["xf"])

V+= [Xs[-1]]

# Bounds for final t
q_lb  = DM([0.1 ,0, 0.1, 0, L])
q_ub  = DM([0.1, 0, 0.1, 0, L])
dq_lb = DM([0]*5)
dq_ub = DM([0]*5)
x_lb = casadi_vec(dae_x, -inf, q=q_lb, dq=dq_lb)
x_ub = casadi_vec(dae_x, inf, q=q_ub, dq=dq_ub)
lbx.append(x_lb)
ubx.append(x_ub)

# Construct regularisation
reg = 0
for x in Xs:
    xstruct = casadi_vec2struct(dae_x,x)
    reg += sumRows(sumCols(xstruct["dq"]**2))
  
for u in Us:
    reg += 100 * sumRows(sumCols(u**2))

e = casadi.vertcat(Us)

# Defining the NLP problem
nlp = {"x": casadi.veccat(V), 
       "f": casadi.dot(e,e) + reg,
       "g": casadi.vertcat([invariants([Xs[0]])[0]] + g)}

# Declaring the NLP solver
solver = nlpsol("solver", "ipopt", nlp)

# Initializationg of the warm point
x0 = casadi.vertcat([x0_guess, u_guess]*N + [x0_guess])

args = {}
args["x0"] = x0
args["lbx"] = casadi.vertcat(lbx)
args["ubx"] = casadi.vertcat(ubx)
args["lbg"] = 0
args["ubg"] = 0

print vertcat(lbx)

res = solver(args)
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
