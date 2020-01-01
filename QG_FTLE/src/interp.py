"""
Module for interpolation and RK4 processing
"""
import numpy as np 
from scipy.interpolate import RegularGridInterpolator
import util 

def calculate_velocity_field( sf ):
    """
    Calculate discretized velocity fields from discretized streamfunction field  
    """
    print('Calculating vloeicty fields...')
    
    xct, yct = sf.shape[0], sf.shape[1]
    reduced_xct, reduced_yct = util.calculate_xct_yct_ratio( xct, yct )
    

    dx = float(reduced_xct)/float(xct)
    dy = float(reduced_yct)/float(yct)
    # import pdb;pdb.set_trace()
    u = np.zeros( np.shape(sf) ) 
    v = np.zeros( np.shape(sf) )

    # TODO: is there some way to vectorize/optimize this?
    for t in range(np.shape(sf)[2]):
        for i in range(1, np.shape(sf)[0]-1): 
            for j in range(1, np.shape(sf)[1]-1):
                u[i,j,t] = (sf[i,j+1,t]-sf[i,j-1,t])/(2*dx)
                v[i,j,t] = (sf[i+1,j,t]-sf[i-1,j,t])/(2*dy)
    print(u.shape)
    print('FInished calculating velcotiy fields')
    return u, v


def calculate_interp_velocity_funcs(u,v): 
    """dg_uinterp
    Calculate interpolation functions based on discretized velocity fields u, v
    rtype: functions vxfunc, vyfunc
    """
    
    xct, yct, ts_max = u.shape[0], u.shape[1], u.shape[2]
    reduced_xct, reduced_yct = util.calculate_xct_yct_ratio( xct, yct )
    dy = dx = float(min( reduced_xct, reduced_yct )) / (float( min( xct, yct ) ))
    # create meshgrid of actual x, y values 
    # note: these values are the center of the cells, so must be shifted by dx and dy 
    _xarr = np.linspace(0.5*dx,float(reduced_xct)-(0.5*dx),xct )
    _yarr = np.linspace(0.5*dy,float(reduced_yct)-(0.5*dy),yct )
    
    vxfunc = [RegularGridInterpolator((_xarr, _yarr), u[:,:,i], method='linear', bounds_error=False) for i in range(ts_max)]
    vyfunc = [RegularGridInterpolator((_xarr, _yarr), v[:,:,i], method='linear', bounds_error=False) for i in range(ts_max)]
    return vxfunc, vyfunc       
    # vxfunc = RegularGridInterpolator((_xarr, _yarr), self.u[:,:,i], method='linear')
    # vyfunc = RegularGridInterpolator((_xarr, _yarr), self.v[:,:,i], method='linear')
    # x, y = np.clip(x,self.minx,self.maxx), np.clip(y,self.miny,self.maxy)
    
    # points = np.transpose(np.array([x, y]))
    # outxv, outxy = vxfunc(points), vyfunc(points)
    # return outxv, outxy


"""
TODO: 
NEED TO FIX UPDATE() 
integrate calculate_interp_velocity_funcs()
"""


# def estimatedVelocity(x,y,u,v,t):
#     minx, maxx = 0.0, 1.0
#     miny, maxy = 0.0, 2.0
#     xct, yct = 64, 128
#     _xarr = np.linspace(minx,maxx,xct)
#     _yarr = np.linspace(miny,maxy,yct)
#     i = int(t/dt)  # get index of time t
#     print(i)
#     vxfunc = RegularGridInterpolator((_xarr, _yarr), -u[:,:,i], method='linear')
#     vyfunc = RegularGridInterpolator((_xarr, _yarr), v[:,:,i], method='linear')
#     x, y = np.clip(x,minx,maxx), np.clip(y,miny,maxy)
#     points = np.transpose(np.array([x, y]))
#     outxv, outxy = vxfunc(points), vyfunc(points)
#     return outxv, outxy


def velocity_update(vfuncs, state, t, dt, x_range=(0.0,2.0), y_range=(0.0,1.0)): 
    """
    computes velocity of particle at state and time
    vfuncs: tuple of list of interp velocity field function objects indexed at ith time step 
    rtype: 
    """
    x = state[:,0]
    y = state[:,1] 
    x, y = np.clip(x, x_range[0], x_range[1]), np.clip(y, y_range[0], y_range[1] )
    points = np.transpose(np.array([x, y]))
    
    i = int(t/dt)  # get index of time t
    vxfunc, vyfunc = vfuncs[0][i], vfuncs[1][i]
    vx, vy = vxfunc(points), vyfunc(points)
    return np.column_stack((-vx,vy))


# # function that computes velocity of particle at each point
# def update(state,t):
#     x = state[:,0]
#     y = state[:,1] 
#     vx, vy = estimatedVelocity(x,y,u,v,t)
#     return np.column_stack((vx,vy))
    
def rk4(vfuncs, state_t, t, dt=0.1, x_range=(0.0,2.0), y_range=(0.0,1.0)):
    tmp_state = state_t
    k1 = dt*velocity_update(vfuncs, tmp_state, t, dt, x_range, y_range )
    k2 = dt*velocity_update(vfuncs, tmp_state+0.5*k1, t+0.5*dt, dt,  x_range, y_range )
    k3 = dt*velocity_update(vfuncs, tmp_state+0.5*k2, t+0.5*dt, dt, x_range, y_range )
    k4 = dt*velocity_update(vfuncs, tmp_state+k3, t+dt, x_range, dt, y_range )
    tmp_state += (k1+2*k2+2*k3+k4)/6
    state = np.zeros(2)
    state[0] = np.clip(tmp_state[0], x_range[0], x_range[1] )
    state[1] = np.clip(tmp_state[1], y_range[0], y_range[1] )
    #noise = B*np.random.normal(u,sigma,(L,2))
    #state[:,4:6] += noise
    return state



