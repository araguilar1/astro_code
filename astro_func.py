"""
Script containing functions for astrodynamic applications

Author: Abram Aguilar

12/29/2019: Creation
"""

import numpy as np
from scipy import constants
import pandas as pd


class CelBod:
    """
    Class for extracting celestial body data used in the CR3BP. The data table below contains the following
    information:

        data[bID][0] = Axial Rotation Period (Rev/Day)
        data[bID][1] = Equatorial Radius (km)
        data[bID][2] = Gravitational Parameter (mu = G*m (km^3/s^2) )
        data[bID][3] = Semi-major Axis of Orbit (km)
        data[bID][4] = Orbital Period (days)
        data[bID][5] = Orbital Eccentricity
        data[bID][6] = Inclination of Orbit to Ecliptic (deg)

        Where bID = Body ID # specified by an integer
    """
    raw_data = [[0.039401, 695700.0, 132712440041.94, 'NA', 'NA', 'NA', 'NA'],  # Sun
                [0.0366, 1738.0, 4902.8, 384400.0, 27.32, 0.0549, 5.145],  # Moon (about Earth)
                [0.017051, 2440.0, 22031.78, 4902.8, 87.97, 0.205647, 7.0039],  # Mercury
                [0.004115, 6051.89, 324858.59, 108208441.28, 224.7, 0.006794, 3.39449],  # Venus
                [1.002737, 6378.14, 398600.44, 149657721.28, 365.26, 0.016192, 0.0045],  # Earth
                [0.974700, 3394.0, 42828.38, 227937168.37, 686.98, 0.093343, 1.84814],  # Mars
                [2.418111, 71492.0, 126686534.91, 778350260.22, 4332.59, 0.048708, 1.30394],  # Jupiter
                [2.252205, 60268.0, 37931207.80, 1433203470.67, 10755.70, 0.050663, 2.48560],  # Saturn
                [-1.3921089, 25559.0, 5793951.32, 2863429878.70, 30685.40, 0.048551, 0.77151],  # Uranus
                [1.489754, 24766.0, 6835099.50, 4501859020.15, 60189.0, 0.007183, 1.77740],  # Neptune
                [0.156563, 1188.30, 869.34, 6018076570.89, 91101.50, 0.258313, 17.22524],  # Pluto
                [0.15625, 605.0, 102.27, 19596.84, 6.39, 0.00005, 112.89596],  # Charon (about Pluto)
                [0.5291, 43.33, 0.0003, 48690.0, 24.85, 0.238214, 112.88839],  # Nix (about Pluto)
                [2.328, 65.0, 0.000320, 64738.0, 38.2, 0.0058652, 0.24200],  # Hydra (about Pluto)
                ['Sync', 2634.0, 9891.0, 1070042.8, 7.15, 0.0006, 0.186],  # Ganymede (about Jupiter)
                ['Sync', 2575.5, 8978.13, 1221870.0, 15.95, 0.0288, 0.28],  # Titan (about Jupiter)
                ['Sync', 788.9, 235.4, 435800.0, 8.71, 0.0022, 0.1],  # Titania (about Uranus)
                [2.644860, 469.7, 62.63, 413968739.37, 1680.22, 0.076103, 10.6007],  # Ceres 
                ['Sync', 252.30, 1.21135, 238040.0, 1.370218, 0.0047, 0.009],  # Enceladus (about Saturn)
                ['Sync', 13.10, 0.000721, 9377.20, 0.32, 0.0151, 1.082],  # Phobos (about Mars)
                ['Sync', 1352.60, 1432.93, 354760.0, 5.876854, 0.000016, 156.834],  # Triton (about Neptune)
                [0.05992, 2403.0, 7181.32, 1883000.0, 16.69, 0.007, 0.281],  # Callisto (about Jupiter)
                []  # Europa (about Jupiter)
                ]

    def __init__(self, name):
        """ Initialize the class """
        self.bName = name
        if name == 'Sun':
            self.bID = 0
        elif name == 'Moon':
            self.bID = 1
        elif name == 'Mercury':
            self.bID = 2
        elif name == 'Venus':
            self.bID = 3
        elif name == 'Earth':
            self.bID = 4
        elif name == 'Mars':
            self.bID = 5
        elif name == 'Jupiter':
            self.bID = 6
        elif name == 'Saturn':
            self.bID = 7
        elif name == 'Uranus':
            self.bID = 8
        elif name == 'Neptune':
            self.bID = 9
        elif name == 'Pluto':
            self.bID = 10
        elif name == 'Charon':
            self.bID = 11
        elif name == 'Nix':
            self.bID = 12
        elif name == 'Hydra':
            self.bID = 13
        elif name == 'Ganymede':
            self.bID = 14
        elif name == 'Titan':
            self.bID = 15
        elif name == 'Titania':
            self.bID = 16
        elif name == 'Ceres':
            self.bID = 17
        elif name == 'Enceladus':
            self.bID = 18
        elif name == 'Phobos':
            self.bID = 19
        elif name == 'Triton':
            self.bID = 20
        elif name == 'Callisto':
            self.bID = 21
        elif name == 'Europa':
            self.bID = 22
        else:
            print('Error: Celestial Body not found')
            self.bID = -1

        self.data = self.raw_data[self.bID]

    def axial_rotation_period(self):
        return self.data[0]

    def equatorial_radius(self):
        return self.data[1]

    def mu(self):
        return self.data[2]

    def orbit_sma(self):
        return self.data[3]

    def orbital_period(self):
        return self.data[4]

    def orbital_ecc(self):
        return self.data[5]

    def orbital_inc(self):
        return self.data[6]

    def mass(self):
        return self.data[2] / (constants.G / 1000 / 1000 / 1000)


def characteristic(primary, secondary):
    """
   Returns array of characteristic quantities of the CR3BP for a primary-secondary system:
   1. Characteristic Length
   2. System Mass Ratio (mu)
   3. Characteristic Time
   """
    G = constants.G / 1000 / 1000 / 1000  # convert from m to km
    l_star = secondary.orbit_sma()
    m1 = primary.mass()
    m2 = secondary.mass()
    m_star = m1 + m2
    mu = m2 / m_star
    t_star = np.sqrt(l_star ** 3 / G / m_star)

    return np.array([l_star, mu, t_star])


def cr3bp_ode(y_, t, mu):
    yd_ = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # Position
    x = y_[0]
    y = y_[1]
    z = y_[2]
    # Velocity
    vx = y_[3]
    vy = y_[4]
    vz = y_[5]

    # Eq. of Motion
    d = np.sqrt((x + mu) ** 2 + y ** 2 + z ** 2)
    r = np.sqrt((x + mu - 1) ** 2 + y ** 2 + z ** 2)

    # Velocity
    yd_[0] = y_[3]
    yd_[1] = y_[4]
    yd_[2] = y_[5]

    # Acceleration
    yd_[3] = -(1 - mu) * (x + mu) / (d ** 3) - mu * (x - 1 + mu) / (r ** 3) + 2 * vy + x
    yd_[4] = -(1 - mu) * y / (d ** 3) - mu * y / (r ** 3) - 2 * vx + y
    yd_[5] = -(1 - mu) * z / (d ** 3) - mu * z / (r ** 3)

    return yd_

def cr3bp_ode_STM(y_, t, mu):
    
    
    ## Unpack the state vector

    # Position
    x = y_[0]
    y = y_[1]
    z = y_[2]

    # Velocity
    xDot = y_[3]
    yDot = y_[4]
    zDot = y_[5]

    # STM
    phi = np.reshape(y_[6:], (6, 6))

    # Scalar Distances from P1 to P3, and P2 to P3
    d = np.sqrt((x + mu) ** 2 + y ** 2 + z ** 2)
    r = np.sqrt((x + mu - 1) ** 2 + y ** 2 + z ** 2)

    d3, d5 = d**3, d**5
    r3, r5 = r**3, r**5

    # Pseudo-Potential Function Partials
    sigXX = 1 - (1-mu)/d3 - mu/r3 + 3*(1-mu)*(x+mu)**2/d5 + 3*mu*(x-1+mu)**2/r5
    sigYY = 1 - (1-mu)/d3 - mu/r3 + 3*(1-mu)*y**2/d5 + 3*mu*y**2/r5
    sigZZ = -(1-mu)/d3 - mu/r3 + 3*(1-mu)*z**2/d5 + 3*mu*z**2/r5
    sigXY = 3*(1-mu)*(x+mu)*y/d5 + 3*mu*(x-1+mu)*y/r5
    sigXZ = 3*(1-mu)*(x+mu)*z/d5 + 3*mu*(x-1+mu)*z/r5
    sigYZ = 3*(1-mu)*y*z/d5 + 3*mu*y*z/r5

    # Jacobian
    A = np.array([ [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                 [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                 [sigXX, sigXY, sigXZ, 0.0, 2.0, 0.0],
                 [sigXY, sigYY, sigYZ, -2.0, 0.0, 0.0],
                 [sigXZ, sigYZ, sigZZ, 0.0, 0.0, 0.0] ])

    # Acceleration 
    xDotDot = -(1 - mu) * (x + mu) / d3 - mu * (x - 1 + mu) / r3 + 2 * yDot + x
    yDotDot = -(1 - mu) * y / d3 - mu * y / r3 - 2 * xDot + y
    zDotDot = -(1 - mu) * z / d3 - mu * z / r3

    yd = np.array([xDot, yDot, zDot, xDotDot, yDotDot, zDotDot])
    phiDotVec = np.dot(A, phi).reshape(36)
    yd = np.hstack((yd, phiDotVec))
    return yd

def ustar(x_, mu):
    """ Returns the pseudo-potential of the CR3BP """
    x = x_[0]
    y = x_[1]
    z = x_[2]
    d = np.sqrt((x + mu) ** 2 + y ** 2 + z ** 2)
    r = np.sqrt((x + mu - 1) ** 2 + y ** 2 + z ** 2)
    u_star = (1 - mu) / d + mu / r + (x ** 2 + y ** 2) / 2
    return u_star


def jacobi(r, mu):
    """Returns the Jacobi constant for some coordinates and mass ratio"""
    uStar = ustar(r[0:3], mu)
    J = 2 * uStar - np.dot(r[3:6], r[3:6])
    # J = 2 - uStar - np.linalg.norm(r[3:6]) ** 2 (Also works, just to show different methods)
    return J


def lagrange_points(mu):
    """Returns the x,y coordinates of the 5 Lagrange points for any system"""
    tol = 1e-12

    def colin(x):
        d = abs(x + mu)
        r = abs(x - 1 + mu)
        a = x
        b = (1 - mu) * (x + mu) / (d ** 3)
        c = mu * (x - 1 + mu) / (r ** 3)
        return a - b - c

    def fd_colin(x):
        xm = x - tol
        xp = x + tol
        ym = colin(xm)
        y = colin(x)
        yp = colin(xp)
        dm = (y - ym) / (x - xm)
        dp = (yp - y) / (xp - x)
        return (dm + dp) / 2

    # L1
    def f1(gam):
        x = 1 - mu - gam
        return colin(x)

    def f1p(gam):
        x = 1 - mu - gam
        return -fd_colin(x)

    gam1 = mu
    delta = 100
    while (abs(delta) > tol):
        delta = f1(gam1) / f1p(gam1)
        gam1 -= delta
    x1 = 1 - mu - gam1

    # L2
    def f2(gam):
        x = 1 - mu + gam
        return colin(x)

    def f2p(gam):
        x = 1 - mu + gam
        return fd_colin(x)

    gam2 = mu
    delta = 100
    while (abs(delta) > tol):
        delta = f2(gam2) / f2p(gam2)
        gam2 -= delta
    x2 = 1 - mu + gam2

    # L3
    def f3(gam):
        x = -mu - gam
        return colin(x)

    def f3p(gam):
        x = -mu - gam
        return -fd_colin(x)

    gam3 = 1
    delta = 100
    while (abs(delta) > tol):
        delta = f3(gam3) / f3p(gam3)
        gam3 -= delta
    x3 = -mu - gam3

    # L4/L5
    x45 = 0.5 - mu
    y4 = np.sqrt(3) / 2
    y5 = -np.sqrt(3) / 2

    return np.array([[x1, 0], [x2, 0], [x3, 0], [x45, y4], [x45, y5]])


def d_2_nd(state, charL, charT, mu):
    """Converts a dimensional state (km, km/s) in the MJ2000 Earth Equator Frame
        to the non-dimensional CR3BP rotating frame"""
    state_nd = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    state_nd[0:3] = state[0:3] / charL
    state_nd[3:6] = charT * state[3:6] / charL
    state_nd[0] = state[0] - mu
    return state_nd


def nd_2_d(state, charL, charT, mu):
    """Converts a non-dimensional CR3BP rotating frame state to the MJ2000
        Earth Equator frame (km, km/s)"""
    state_d = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    state_d[0] = state[0] + mu
    state_d[0:3] = charL * state[0:3]
    state_d[3:6] = charL * state[3:6] / charT
    return state_d


def per_orb_df():
    """Function returns a dataframe of period Lagrange Orbit information created by Dan Grebow.
        No input is required."""
    data = r'periodicLagrangeOrbits.csv'
    df = pd.read_csv(data)
    return df
