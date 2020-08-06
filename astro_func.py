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
                [0.0366, 1738.0, 4902.8, 384400.0, 27.32, 0.0549, 5.145],  # Moon
                [0.017051, 2440.0, 22031.78, 4902.8, 87.97, 0.205647, 7.0039],  # Mercury
                [0.004115, 6051.89, 324858.59, 108208441.28, 224.7, 0.006794, 3.39449],  # Venus
                [1.002737, 6378.14, 398600.44, 149657721.28, 365.26, 0.016192, 0.0045],  # Earth
                [0.974700, 3394.0, 42828.38, 227937168.37, 686.98, 0.093343, 1.84814],  # Mars
                [],  # Jupiter
                [],  # Saturn
                [],  # Uranus
                [],  # Neptune
                [],  # Pluto
                [],  # Charon
                [],  # Nix
                [],  # Hydra
                [],  # Ganymede
                [],  # Titan
                [],  # Titania
                [],  # Ceres
                [],  # Enceladus
                [],  # Phobos
                [],  # Triton
                [],  # Callisto
                []  # Europa
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
    state[0:2] = state[0:2] / charL
    state[3:5] = charT * state[3:5] / charL
    state[0] = state[0] - mu
    return state


def nd_2_d(state, charL, charT, mu):
    """Converts a non-dimensional CR3BP rotating frame state to the MJ2000
        Earth Equator frame (km, km/s)"""
    state[0] = state[0] + mu
    state[0:2] = charL * state[0:2]
    state[3:5] = charL * state[3:5] / charT
    return state


def per_orb_df():
    """Function returns a dataframe of period Lagrange Orbit information created by Dan Grebow.
        No input is required."""
    data = r'periodicLagrangeOrbits.csv'
    df = pd.read_csv(data)
    return df
