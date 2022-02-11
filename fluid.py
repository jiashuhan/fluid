"""
----------------------------------------------------------
2D Finite Volume simulation of an ideal compressible fluid
----------------------------------------------------------
"""
import sys, numpy as np, matplotlib.pyplot as plt
import matplotlib.animation as animation

def prim2cons(rho, vx, vy, p, gamma, V):
    """
    Convert primitive/convective variables to conservative variables.

    Parameters
    ----------
    rho, vx, vy, p: array_like, float
        cell density, velocities, and pressure

    gamma: float
        ideal gas adiabatic index

    V: float
        cell volume

    Returns
    -------
    m, px, py, E: array_like, float
        cell mass, momenta, and energy
    """
    m = rho * V
    px = m * vx
    py = m * vy
    E = (p/(gamma-1) + rho*(vx**2+vy**2)/2) * V
    return m, px, py, E

def cons2prim(m, px, py, E, gamma, V):
    """
    Convert conservative variables to primitive/convective variables.

    Parameters
    ----------
    m, px, py, E: array_like, float
        cell mass, momenta, and energy

    gamma: float
        ideal gas adiabatic index

    V: float
        cell volume

    Returns
    -------
    rho, vx, vy, p: array_like, float
        cell density, velocities, and pressure
    """
    rho = m / V
    vx = px / m
    vy = py / m
    p = (gamma-1)*(E/V-rho*(vx**2+vy**2)/2)
    return rho, vx, vy, p

def gradient(f, dx):
    """
    Calculate the gradient of a field. Assumes periodic boundary conditions.

    Parameters
    ----------
    f: array_like, float
        array representing the field 

    dx: float
        cell size

    Returns 
    -------
    f_dx, f_dy: array_like, float
        arrays representing the derivatives
    """
    f_dx = (np.roll(f,-1,axis=0)-np.roll(f,1,axis=0))/2/dx
    f_dy = (np.roll(f,-1,axis=1)-np.roll(f,1,axis=1))/2/dx
    return f_dx, f_dy

def extrapolate(f, f_dx, f_dy, dx):
    """
    Spatially extrapolate the values at cell faces using gradients.

    Paramaters
    ----------
    f, f_dx, f_dy: array_like, float
        arrays of the field and its gradient

    dx: float
        cell size

    Returns
    -------
    fxL, fxR, fyL, fyR: array_like, float
        arrays of spatially extrapolated values on the cell faces

    Note
    ----
    After shifting the L arrays by 1 to the left, the L arrays represent
    the R side of the R face of every cell, whereas the R arrays represent 
    the L side of the R face of every cell.
    """
    # shift the L face of cell 1 to the left to match it with the R face of cell 0
    fxL = np.roll(f-f_dx*dx/2, -1, axis=0)
    fxR = f+f_dx*dx/2
    fyL = np.roll(f-f_dy*dx/2, -1, axis=1)
    fyR = f + f_dy*dx/2
    return fxL, fxR, fyL, fyR

def rusanov_flux(rhoL, rhoR, vxL, vxR, vyL, vyR, pL, pR, gamma):
    """
    Calculates the flux between two states (time steps).

    Parameters
    ----------
    rhoL, rhoR, vxL, vxR, vyL, vyR, pL, pR: array_like, float
        arrays for the primitive variables on the left side and r
        ight side of the interfaces, respectively

    gamma: float
        ideal gas adiabatic index

    Returns
    -------
    Fm, Fpx, Fpy, FE: array_like, float
        arrays representing the fluxes of each of the conservative variables

    Note
    ----
    For vertical (horizontal) interfaces, the y (x)-fluxes are zero, since 
    dx (dy) across the interface is zero. For fluxes in the other direction,
    swap x <-> y in the parameters and returned variables.
    """
    # energy per unit volume
    eL = pL/(gamma-1)+rhoL*(vxL**2+vyL**2)/2
    eR = pR/(gamma-1)+rhoL*(vxR**2+vyR**2)/2

    pxL = rhoL*vxL
    pxR = rhoR*vxR
    pyL = rhoL*vyL
    pyR = rhoR*vyR
    
    FmL = pxL
    FmR = pxR
    FpxL = pxL**2/rhoL + pL
    FpxR = pxR**2/rhoR + pR
    FpyL = pxL*pyL/rhoL
    FpyR = pxR*pyR/rhoR
    FEL = (eL+pL)*pxL/rhoL
    FER = (eR+pR)*pxR/rhoR

    # take the average between the two sides
    Fm = (FmL+FmR)/2
    Fpx = (FpxL+FpxR)/2
    Fpy = (FpyL+FpyR)/2
    FE = (FEL+FER)/2

    # wavespeeds
    cL = np.sqrt(gamma*pL/rhoL)+np.abs(vxL)
    cR = np.sqrt(gamma*pR/rhoR)+np.abs(vxR)
    c = np.maximum(cL, cR)

    # add numerical diffisivity term which keeps solution stable
    # note that R(L) actually corresponds to the L(R) side of each interface
    Fm -= c*(rhoL-rhoR)/2
    Fpx -= c*(rhoL*vxL-rhoR*vxR)/2
    Fpy -= c*(rhoL*vyL-rhoR*vyR)/2
    FE -= c*(eL-eR)/2
    return Fm, Fpx, Fpy, FE

def update_cells(f, Fx, Fy, dx, dt):
    """
    Apply flux to the conservative variable f.

    Parameters
    ----------
    f: array_like, float
        the conservative variable to be updated

    Fx, Fy: array_like, float
        the fluxes in the x and y directions, respectively

    dx: float
        cell size

    dt: float
        time step

    Note
    ----
    We multiply by dx here instead of divide because we are actually
    working with the cell mass/momentum/energy, which have an extra
    factor of V=dx*dy=dx^2.
    """
    # update the L side of the R face along x
    f += -dt*dx*Fx
    # update the R side of the R face along x
    f += dt*dx*np.roll(Fx,1,axis=0)
    # update the L side of the R face along y
    f += -dt*dx*Fy
    # update the R side of the R face along y
    f += dt*dx*np.roll(Fy,1,axis=1)
    return f

def main(m, px, py, E, dx, gamma, duration=2):
    """
    Main loop of simulation

    Parameters
    ----------
    m, px, py, E: array_like, float
        cell mass, momenta, and energy

    dx: float
        cell size

    Returns
    -------
    frames: list
        list of cell densities in all frames.
    """
    frames = []

    cfl_factor = 0.4 # Courant-Friedrichs-Lewy factor
    t_plot = 0.02 # plot frequency
    V = dx**2
    
    fig = plt.figure(figsize=(4,4), dpi=100)
    N_frames = 1 # number of frames
    t = 0
    while t < duration:
        rho, vx, vy, p = cons2prim(m, px, py, E, gamma, V)

        # time step = (CFL factor) * dx / (maximum signal speed)
        dt = cfl_factor * np.min(dx/(np.sqrt(gamma*p/rho)+np.sqrt(vx**2+vy**2)))
        # somehow rho and p have negative values

        plot_now = False
        if t+dt > N_frames*t_plot:
            dt = N_frames*t_plot-t # sync next step with the next frame 
            plot_now = True

        rho_dx, rho_dy = gradient(rho, dx)
        vx_dx, vx_dy = gradient(vx, dx)
        vy_dx, vy_dy = gradient(vy, dx)
        p_dx, p_dy = gradient(p, dx)

        # move half a time step forward for spatial extrapolations and flux calculations
        rho1 = rho - dt*(vx*rho_dx+vy*rho_dy+rho*(vx_dx+vy_dy))/2 
        vx1 = vx - dt*(vx*vx_dx+vy*vx_dy+p_dx/rho)/2
        vy1 = vy - dt*(vx*vy_dx+vy*vy_dy+p_dy/rho)/2
        p1 = p - dt*(vx*p_dx+vy*p_dy+gamma*p*(vx_dx+vy_dy))/2

        # spatially extrapolate for field values at interfaces
        rho_xL, rho_xR, rho_yL, rho_yR = extrapolate(rho1, rho_dx, rho_dy, dx)
        vx_xL, vx_xR, vx_yL, vx_yR = extrapolate(vx1, vx_dx, vx_dy, dx)
        vy_xL, vy_xR, vy_yL, vy_yR = extrapolate(vy1, vy_dx, vy_dy, dx)
        p_xL, p_xR, p_yL, p_yR = extrapolate(p1, p_dx, p_dy, dx)
        
        # calculate fluxes across vertical (x) faces
        Fmx, Fpxx, Fpyx, FEx = rusanov_flux(rho_xL, rho_xR, vx_xL, vx_xR, vy_xL, vy_xR, p_xL, p_xR, gamma)
        # calculate fluxes across horizontal (y) faces (swap all x <-> y)
        Fmy, Fpyy, Fpxy, FEy = rusanov_flux(rho_yL, rho_yR, vy_yL, vy_yR, vx_yL, vx_yR, p_yL, p_yR, gamma)
        
        # update cells
        m = update_cells(m, Fmx, Fmy, dx, dt)
        px = update_cells(px, Fpxx, Fpxy, dx, dt)
        py = update_cells(py, Fpyx, Fpyy, dx, dt)
        E = update_cells(E, FEx, FEy, dx, dt)

        t += dt

        if plot_now or (t >= duration):
            plt.cla() # clear the axes
            plt.imshow(rho.T, cmap='turbo')
            ax = plt.gca()
            ax.invert_yaxis()
            ax.set_aspect('equal')
            plt.pause(0.001)
            N_frames += 1
        
        frames.append(rho.T)

    plt.savefig('./results/final_state.png', dpi=240)
    plt.show()
    return frames

def kh_init(N, gamma=5/3, box_size=1):
    """
    Generate initial conditions for Kelvin-Helmholtz instabilty.
    
    Parameters
    ----------
    N: int
        spatial resolution (number of cells in each direction)

    Returns
    -------
    (m, px, py, E, dx, gamma): tuple
        arrays representing the cell mass, momenta, and energy, 
        as well as the cell size and gamm

    Note
    ----
    A high density region moves rightward in a left-moving background,
    with uniform pressure. The instability is induced by a small 
    velocity perturbation perpendicular to the shear surface.
    """
    dx = box_size/N
    x = np.linspace(dx/2, box_size-dx/2, N)
    Y, X = np.meshgrid(x, x)
    # np.abs(Y-box_size/2) creates a configuration symmetric about the x axis
    # with the center being 0 and the edge being box_size/2
    rho = 1. + (np.abs(Y-box_size/2) < box_size/4) # a central band of width box_size/2 and value 2
    vx = -0.5 + (np.abs(Y-box_size/2) < box_size/4) # central band with vx = 0.5; outside with vx = -0.5
    # alternating vy perturbations along narrow Gaussian bands of width sigma at the flow boundaries
    sigma = 0.03 # width of perturbation along y-direction
    vy = 0.1*np.sin(4*np.pi*X)*(np.exp(-(Y-box_size/4)**2/(2*sigma**2))+np.exp(-(Y-3*box_size/4)**2/(2*sigma**2)))
    p = 2.5*np.ones(X.shape) # uniform pressure

    m, px, py, E = prim2cons(rho, vx, vy, p, gamma, dx**2)
    return m, px, py, E, dx, gamma

def init(N, gamma=5/3, box_size=1):
    """ Create initial conditions """
    dx = box_size/N
    x = np.linspace(dx/2, box_size-dx/2, N)
    Y, X = np.meshgrid(x, x)
    rho = 1+(2+(np.abs(Y-box_size/2) < 0.2) + (np.abs(X-box_size/2) < 0.2))%2 # checkerboard pattern
    vx = -10*((np.abs(Y-box_size/2) < 0.2)+(np.abs(X-box_size/2) > 0.2)-1)*(np.abs(Y-box_size/2)-box_size/2)*(((Y-box_size/2) > 0)-0.5)
    vy = 10*((np.abs(X-box_size/2) < 0.2)+(np.abs(Y-box_size/2) > 0.2)-1)*(np.abs(X-box_size/2)-box_size/2)*(((X-box_size/2) > 0)-0.5)
    vx += np.random.normal(loc=0, scale=0.5, size=(N,N)) # add stochastic components to velocities
    vy += np.random.normal(loc=0, scale=0.5, size=(N,N))
    p = np.random.uniform(1, 3, size=(N,N))
    m, px, py, E = prim2cons(rho, vx, vy, p, gamma, dx**2)
    return m, px, py, E, dx, gamma

def animate(i):
    plt.cla()
    plt.imshow(frames[i], cmap='turbo')

if __name__== "__main__":
    #frames = main(*kh_init(128))
    frames = main(*init(128))
    fig = plt.figure(figsize=(4,4), dpi=100)
    anim = animation.FuncAnimation(fig, animate, len(frames), interval=1, blit=False)
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=100)
    anim.save('./results/fluid.mp4', writer=writer)
    plt.show()
