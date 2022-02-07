import numpy as np
from scipy import integrate
from scipy import constants
import pandas as pd
import plotly.io as pio
import plotly.graph_objects as go
import astro_func as af
from plotly.offline import plot

pio.templates.default = 'plotly_dark'
#test
G = constants.G / 1000 / 1000 / 1000  # Convert to km^3 / (kg * s^2)

earth = af.CelBod('Earth')
moon = af.CelBod('Moon')

CQ = af.characteristic(earth, moon)
mu = CQ[0]
charL = CQ[1]
charT = CQ[2]

df = af.per_orb_df() # DataFrame of Periodic Orbits, use as wanted

def main():
    
    y0 = np.array([1.07597327, 0.0, -.2022, 0.0, -.193298067, 0.0])

    phi0 = np.eye(6).reshape(36)

    y0_STM = np.hstack((y0,phi0))  # Use this for STM ode

    nPrimOrb = 1
    tf = nPrimOrb * 2 * np.pi
    EaMn_lag = af.lagrange_points(mu)

    ODE = lambda t, y_: af.cr3bp_ode(y_, t, mu)         # Use this for no STM 
    ODE_STM = lambda t, y_: af.cr3bp_ode_STM(y_, t, mu) # Use This for STM

    sol = integrate.solve_ivp(ODE_STM, [0, tf], y0_STM, method='DOP853', atol=1e-12, rtol=1e-12)
    
    traj = sol.y
    t = sol.t

    # Trajectory
    x = traj[0, :]
    y = traj[1, :]
    z = traj[2, :]
    xdot = traj[3, :]
    ydot = traj[4, :]
    zdot = traj[5, :]
    # Monodromy Matrix
    phi = traj[6:42, -1]
    phi = phi.reshape(6, 6)

    jacobi = af.jacobi(y0, mu)
    print('\nJacobi for Trajectory:', jacobi, '\n')

    # Jacobi for each lagrange point
    lagJacZV = np.array([])
    for i in range(5):
        xl = EaMn_lag[i, 0]
        yl = EaMn_lag[i, 1]
        rl = np.array([xl, yl, 0., 0., 0., 0.])
        j = af.jacobi(rl, mu)
        lagJacZV = np.append(lagJacZV, j)
        print('ZV for L' + str(i+1), j)

    # Jacobi for entire Trajectory
    jacTraj = np.array([])
    for i in range(np.shape(traj)[1]):
        r1 = np.array([x[i], y[i], z[i], xdot[i], ydot[i], zdot[i]])
        jacTraj = np.append(jacTraj, af.jacobi(r1, mu))

    print('\nNon-dimensional state:', y0, '(ND)\n')
    state = af.nd_2_d(y0, charL, charT, mu)
    print('Dimensional State:', state, '(km, km/s)\n')

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines', line=dict(color='pink', width=2),
                               name='Trajectory'))
    fig.add_trace(go.Scatter3d(x=[1 - mu], y=[0], z=[0], mode='markers', marker=dict(size=6, color='lightgray'),
                               name='Moon'))
    fig.add_trace(go.Scatter3d(x=[y0[0]], y=[y0[1]], z=[y0[2]], mode='markers', marker=dict(size=3, color='red'),
                               name='ic'))
    # Lagrange points
    fig.add_trace(go.Scatter3d(x=[EaMn_lag[0, 0]], y=[EaMn_lag[0, 1]], z=[0], mode='markers',
                               marker=dict(size=3.5, color='green', symbol='x'), name='L1'))
    fig.add_trace(go.Scatter3d(x=[EaMn_lag[1, 0]], y=[EaMn_lag[1, 1]], z=[0], mode='markers',
                               marker=dict(size=3.5, color='green', symbol='x'), name='L2'))

    fig.update_layout(scene=dict(xaxis_title='X (ND)', yaxis_title='Y (ND)', zaxis_title='Z (ND)',
                                 ), scene_aspectmode='data', title='Trajectory in Earth-Moon System')

    # camera = dict(eye=dict(x=0, y=1, z=-2))
    # fig.update_layout(scene_camera=camera)

    plot(fig, filename='orbit.html', auto_open=True)

    # Plot Jacobi Const of Trajectory
    fig2 = go.Figure()
    fig2.add_trace(go.Scattergl(x=t, y=jacTraj, name='Jacobi Constant', showlegend=True))
    fig2.update_layout(title='Jacobi Constant for Trajectory', xaxis_title='Time (ND)', yaxis_title='Jacobi Constant')

    plot(fig2, filename='jacobi.html', auto_open=True)

if __name__ == '__main__':
    main()
