import casadi as C
from collections import OrderedDict

def casadi_vec2struct(s,vec):
  try:
      vec.sparsity()
  except:
      vec = C.DM(vec)
  if isinstance(s,OrderedDict):
    out = OrderedDict()
    sizes = [0]
    for f in s.keys():
      n = casadi_struct2vec(s[f]).shape[0]
      sizes.append(sizes[-1]+n)

    comps = C.vertsplit(vec,sizes)
    for i,f in enumerate(s.keys()):
      out[f] = casadi_vec2struct(s[f],comps[i])
    return out
  else:
    return C.reshape(vec,*s.shape)

def casadi_struct2vec(s):
  flat = []
  if isinstance(s,OrderedDict):
    for f in s.keys():
      flat.append(casadi_struct2vec(s[f]))
    return C.vertcat(flat)
  else:
    return C.vec(s)

def casadi_struct(s,default=0,**kwargs):
  ret = OrderedDict()
  args = dict(kwargs)
  for k in s.keys():
    if k in kwargs:
      e = args[k]
      del args[k]
      if not hasattr(e,'shape'):
        e = C.DM(e)
      if e.is_scalar():
        e = C.repmat(e,*s[k].shape)
      assert e.shape == s[k].shape
    else:
      e = default*C.DM.ones(*s[k].shape)
    ret[k] = e
  return ret

def casadi_vec(s,default=0,**kwargs):
  return casadi_struct2vec(casadi_struct(s,default=default,**kwargs))

def skew( w ):
    x = w[0]
    y = w[1]
    z = w[2]
    return C.blockcat([[0,-z,y],[z,0,-x],[-y,x,0]])

