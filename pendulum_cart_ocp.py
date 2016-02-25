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

# order dictionary
dae_x = OrderedDict([('q',q),('dq',dq)])
dae_z = OrderedDict([('z',z)])
dae = {}
dae["x"] = casadi_struct2vec(dae_x)
dae["z"] = casadi_struct2vec(dae_z)
dae["p"] = u
dae["ode"] = casadi_vec(dae_x,q=dq,dq=ddq)
dae["alg"] = vertcat([res_eq, C])

