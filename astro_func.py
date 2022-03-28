"""
Script containing functions for astrodynamic applications

Author: Abram Aguilar

12/29/2019: Creation
"""

import numpy as np
from scipy.constants import G
import pandas as pd
import matplotlib.pyplot as plt

class CelestialObject:
    """
    Class for extracting celestial body data used in the CR3BP. The data table below contains the following
    information:

        data[ID][0] = Axial Rotation Period (Rev/Day)
        data[ID][1] = Equatorial Radius (km)
        data[ID][2] = Gravitational Parameter (mu = G*m (km^3/s^2) )
        data[ID][3] = Semi-major Axis of Orbit (km)
        data[ID][4] = Orbital Period (days)
        data[ID][5] = Orbital Eccentricity
        data[ID][6] = Inclination of Orbit to Ecliptic (deg)

        Where ID = Body ID # specified by an integer
    """
    raw_data = [
                [0.039401, 695700.0, 132712440041.94, '--', '--', '--', '--'],  # Sun
                [0.0366, 1738.0, 4902.8, 384400.0, 27.32, 0.0549, 5.145],  # Moon 
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
                ['Synch', 2634.0, 9891.0, 1070042.8, 7.15, 0.0006, 0.186],  # Ganymede (about Jupiter)
                ['Synch', 2575.5, 8978.13, 1221870.0, 15.95, 0.0288, 0.28],  # Titan (about Jupiter)
                ['Synch', 788.9, 235.4, 435800.0, 8.71, 0.0022, 0.1],  # Titania (about Uranus)
                [2.644860, 469.7, 62.63, 413968739.37, 1680.22, 0.076103, 10.6007],  # Ceres 
                ['Synch', 252.30, 1.21135, 238040.0, 1.370218, 0.0047, 0.009],  # Enceladus (about Saturn)
                ['Synch', 13.10, 0.000721, 9377.20, 0.32, 0.0151, 1.082],  # Phobos (about Mars)
                ['Synch', 1352.60, 1432.93, 354760.0, 5.876854, 0.000016, 156.834],  # Triton (about Neptune)
                [0.05992, 2403.0, 7181.32, 1883000.0, 16.69, 0.007, 0.281],  # Callisto (about Jupiter)
                [4.2625, 151.959, 3203.56, 671100.0, 3.552, 0.0094, 7.483], # Europa (about Jupiter)
                ]

    def __init__(self, name):
        self.body = name
        if self.body.upper() == "SUN":
            self.ID = 0
        elif self.body.upper() == "MOON":
            self.ID = 1
        elif self.body.upper() == "MERCURY":
            self.ID = 2
        elif self.body.upper() == "VENUS":
            self.ID = 3
        elif self.body.upper() == "EARTH":
            self.ID = 4
        elif self.body.upper() == "MARS":
            self.ID = 5
        elif self.body.upper() == "JUPITER":
            self.ID = 6
        elif self.body.upper() == "SATURN":
            self.ID = 7
        elif self.body.upper() == "URANUS":
            self.ID = 8
        elif self.body.upper() == "NEPTUNE":
            self.ID = 9
        elif self.body.upper() == "PLUTO":
            self.ID = 10
        elif self.body.upper() == "CHARON":
            self.ID = 11
        elif self.body.upper() == "NIX":
            self.ID = 12
        elif self.body.upper() == "HYDRA":
            self.ID = 13
        elif self.body.upper() == "GANYMEDE":
            self.ID = 14
        elif self.body.upper() == "TITAN":
            self.ID = 15
        elif self.body.upper() == "TITANIA":
            self.ID = 16
        elif self.body.upper() == "CERES":
            self.ID = 17
        elif self.body.upper() == "ENCELADUS":
            self.ID = 18
        elif self.body.upper() == "PHOBOS":
            self.ID = 19
        elif self.body.upper() == "TRITON":
            self.ID = 20
        elif self.body.upper() == "CALLISTO":
            self.ID = 21
        elif self.body.upper() == "EUROPA":
            self.ID = 22
        else:
            print("Error: Celesital Object not found")
            self.ID = 999999

        self.data = self.raw_data[self.ID]

    def axial_rotation_period(self):
        return self.data[0]

    def equatorial_radius(self):
        return self.data[1]

    def mu(self):
        return self.data[2]

    def mass(self):
        return self.data[2] / (G / 1000**3)

    def orbit_semi_major_axis(self):
        return self.data[3]

    def orbital_period(self,unit="day"):
        if unit == "day":
            return self.data[4]
        elif unit == "year":
            return self.data[4]/365.25
        else:
            print("Warning: Unit not found. Defaulting to days.")
            return self.data[4]

    def orbit_eccentricity(self):
        return self.data[5]

    def orbit_inclination(self):
        return self.data[6]


def CharacteristicQuantities(P1, P2):
    """
   Returns array of characteristic quantities of the CR3BP for a primary-secondary system:
   1. System Mass Ratio (mu) 
   2. Characteristic Length
   3. Characteristic Time
   """
    m_star = P1.mass() + P2.mass()
    m2 = P2.mass()
    mu = m2 / m_star
    lstar = P2.orbit_semi_major_axis()
    tstar = np.sqrt(lstar ** 3 / (G/1000**3) / m_star)
    return [mu, lstar, tstar]


def CR3BP_ODE(ndx, t, mu):
    # Position
    x = ndx[0]
    y = ndx[1]
    z = ndx[2]

    # Velocity
    vx = ndx[3]
    vy = ndx[4]
    vz = ndx[5]

    # Scalar distances from P1 to P3 and P2 to P3 respectively
    d = np.sqrt((x + mu) ** 2 + y ** 2 + z ** 2)
    r = np.sqrt((x + mu - 1) ** 2 + y ** 2 + z ** 2)

    # Acceleration
    ax = -(1 - mu) * (x + mu) / (d ** 3) - mu * (x - 1 + mu) / (r ** 3) + 2 * vy + x
    ay = -(1 - mu) * y / (d ** 3) - mu * y / (r ** 3) - 2 * vx + y
    az = -(1 - mu) * z / (d ** 3) - mu * z / (r ** 3)

    return np.array([vx, vy, vz, ax, ay, az])


def CR3BP_ODE_STM(ndx, t, mu):
    
    # Position
    x = ndx[0]
    y = ndx[1]
    z = ndx[2]

    # Velocity
    vx = ndx[3]
    vy = ndx[4]
    vz = ndx[5]

    # STM
    phi = np.reshape(ndx[6:], (6, 6))

    # Scalar Distances from P1 to P3, and P2 to P3 respectively
    d = np.sqrt((x + mu) ** 2 + y ** 2 + z ** 2)
    r = np.sqrt((x + mu - 1) ** 2 + y ** 2 + z ** 2)

    d3, d5 = d**3, d**5
    r3, r5 = r**3, r**5

    sigXX = 1 - (1-mu)/d3 - mu/r3 + 3*(1-mu)*(x+mu)**2/d5 + 3*mu*(x-1+mu)**2/r5
    sigYY = 1 - (1-mu)/d3 - mu/r3 + 3*(1-mu)*y**2/d5 + 3*mu*y**2/r5
    sigZZ = -(1-mu)/d3 - mu/r3 + 3*(1-mu)*z**2/d5 + 3*mu*z**2/r5
    sigXY = 3*(1-mu)*(x+mu)*y/d5 + 3*mu*(x-1+mu)*y/r5
    sigXZ = 3*(1-mu)*(x+mu)*z/d5 + 3*mu*(x-1+mu)*z/r5
    sigYZ = 3*(1-mu)*y*z/d5 + 3*mu*y*z/r5

    # Jacboian
    A = np.array([ [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                 [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                 [sigXX, sigXY, sigXZ, 0.0, 2.0, 0.0],
                 [sigXY, sigYY, sigYZ, -2.0, 0.0, 0.0],
                 [sigXZ, sigYZ, sigZZ, 0.0, 0.0, 0.0] ])

    # Acceleration 
    ax = -(1 - mu) * (x + mu) / d3 - mu * (x - 1 + mu) / r3 + 2 * vy + x
    ay = -(1 - mu) * y / d3 - mu * y / r3 - 2 * vx + y
    az = -(1 - mu) * z / d3 - mu * z / r3

    yd = np.array([vx, vy, vz, ax, ay, az])
    phiDotVec = np.dot(A, phi).reshape(36) 
    yd = np.hstack((yd, phiDotVec))
    return yd


def uStar(ndx, mu):
    """ Returns the pseudo-potential of the CR3BP """
    x = ndx[0]
    y = ndx[1]
    z = ndx[2]
    d = np.sqrt((x + mu) ** 2 + y ** 2 + z ** 2)
    r = np.sqrt((x + mu - 1) ** 2 + y ** 2 + z ** 2)
    return (1 - mu) / d + mu / r + (x ** 2 + y ** 2) / 2


def jacobi(ndx, mu):
    """Returns the Jacobi constant for some coordinates and mass ratio"""
    ustar = uStar(ndx, mu)
    C = 2 * ustar - np.dot(ndx[3:], ndx[3:])
    # C = 2 - uStar - np.linalg.norm(ndx[3:]) ** 2 (Also works, just to show different methods)
    return C


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


def nondim(x, lstar, tstar, mu):
    """Converts a dimensional state (km, km/s) in the MJ2000 Earth Equator Frame
        to the non-dimensional CR3BP rotating frame"""
    ndx      = np.copy(x)
    ndx[0:3] = ndx[0:3] / lstar
    ndx[3:6] = ndx[3:6] * tstar / lstar
    ndx[0]   = ndx[0] - mu
    return ndx


def dimensionalize(ndx, lstar, tstar, mu):
    """Converts a non-dimensional CR3BP rotating frame state to the MJ2000
        Earth Equator frame (km, km/s)"""
    x      = np.copy(ndx)
    x[0]   = x[0] + mu
    x[0:3] = x[0:3] * lstar
    x[3:6] = x[3:6] * lstar / tstar
    return x


def per_orb_df():
    """Function returns a dataframe of period Lagrange Orbit information created by Dan Grebow.
        No input is required."""
    data = r'periodicLagrangeOrbits.csv'
    df = pd.read_csv(data)
    return df


def plotZVC(mu, C, fill=0):
    """ Plots the ZVC surface for a given mu and Jacobi constant, NOTE: uses matplotlib by default"""
    ax = plt.gca()

    npoints = 1000
    x = np.linspace(-1.5, 1.5, npoints)
    y = np.linspace(0.0, 1.5, npoints)
    ZZ = np.empty((npoints, npoints))
    ZZ[:] = np.NaN
    zzPos = np.zeros((npoints, npoints))
    eps = 0.1
    for i in range(npoints):
        for j in range(npoints):
            ndx = np.array([x[i], y[j], 0.0])
            temp = jacobi(ndx, mu)
            if (C >= temp) and (temp > (C * (1 - eps))):
                ZZ[j, i] = temp
                zzPos[j, i] = 1
    
    if fill == 1:
        ax.contourf(x, y, ZZ)
        ax.contourf(x, -y, ZZ)
    else:
        ax.contour(x, y, ZZ)
        ax.contour(x, -y, ZZ)
