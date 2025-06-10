import math
import numpy as np
from typing import List

def mod2pi(theta):
    # return theta - 2 * math.pi * math.floor((theta + math.pi) / (2 * math.pi))
    return theta % (2 * math.pi)


"""
Classical 2D Dubins Curve
"""
class DubinsStruct:
    def __init__(self, t: float, p: float, q: float, length: float, case: str):
        self.t = t
        self.p = p
        self.q = q
        self.length = length
        self.case = case

    def __repr__(self):
        return (
            f"DubinsStruct("
            f"\n\t\t\t t={self.t},\n\t\t\t p={self.p},\n\t\t\t q={self.q}, "
            f"\n\t\t\t length={self.length},\n\t\t\t case='{self.case}')"
        )


"""
Classical 2D Dubins Curve
"""


class DubinsManeuver2D:
    def __init__(
        self, qi: List[float], qf: List[float], rhomin: float, maneuver: DubinsStruct
    ):
        self.qi = qi
        self.qf = qf
        self.rhomin = rhomin
        self.maneuver = maneuver
    
    def __repr__(self):
        return (
            f"DubinsManeuver2D("
            f"\n\t\t qi={self.qi}, \n\t\t qf={self.qf}, \n\t\t rhomin={self.rhomin}, "
            f"\n\t\t maneuver={self.maneuver})"
        )


class DubinsManeuver3D:
    def __init__(self, qi, qf, rhomin, pitchlims, path, length):
        self.qi = qi
        self.qf = qf
        self.rhomin = rhomin
        self.pitchlims = pitchlims
        self.path = path
        self.length = length

    def __repr__(self):
        return (
            f"DubinsManeuver3D("
            f"\n\t qi={self.qi}, \n\t qf={self.qf}, \n\t rhomin={self.rhomin},"
            f"\n\t pitchlims={self.pitchlims},\n\t path={self.path}, \n\t length={self.length})"
        )

def DubinsManeuver2D_func(
    qi: List[float], qf: List[float], rhomin=1.0, minLength=None, disable_CCC=False
) -> DubinsManeuver2D:
    maneuver = DubinsManeuver2D(
        qi, qf, rhomin, DubinsStruct(0.0, 0.0, 0.0, math.inf, "")
    )

    dx = maneuver.qf[0] - maneuver.qi[0]
    dy = maneuver.qf[1] - maneuver.qi[1]
    D = math.sqrt(dx**2 + dy**2)

    # Distance normalization
    d = D / maneuver.rhomin

    # Normalize the problem using rotation
    rotationAngle = mod2pi(math.atan2(dy, dx))

    a = mod2pi(maneuver.qi[2] - rotationAngle)
    b = mod2pi(maneuver.qf[2] - rotationAngle)

    sa = math.sin(a)
    ca = math.cos(a)
    sb = math.sin(b)
    cb = math.cos(b)

    # CSC
    pathLSL = _LSL(maneuver, a, b, d, sa, ca, sb, cb)
    pathRSR = _RSR(maneuver, a, b, d, sa, ca, sb, cb)
    pathLSR = _LSR(maneuver, a, b, d, sa, ca, sb, cb)
    pathRSL = _RSL(maneuver, a, b, d, sa, ca, sb, cb)

    if disable_CCC:
        _paths = [pathLSL, pathRSR, pathLSR, pathRSL]
    else:
        # CCC
        pathRLR = _RLR(maneuver, a, b, d, sa, ca, sb, cb)
        pathLRL = _LRL(maneuver, a, b, d, sa, ca, sb, cb)
        _paths = [pathLSL, pathRSR, pathLSR, pathRSL, pathRLR, pathLRL]

    if (
        abs(d) < maneuver.rhomin * 1e-5
        and abs(a) < maneuver.rhomin * 1e-5
        and abs(b) < maneuver.rhomin * 1e-5
    ):
        dist_2D = np.max(np.abs(np.array(maneuver.qi[0:2]) - np.array(maneuver.qf[0:2])))
        if dist_2D < maneuver.rhomin * 1e-5:
            pathC = _C(maneuver)
            _paths = [pathC]

    _paths = sorted(_paths, key=lambda x: x.length)

    if minLength is None:
        maneuver.maneuver = _paths[0]
    else:
        for p in _paths:
            if p.length >= minLength:
                maneuver.maneuver = p
                break

        if maneuver.maneuver is None:
            inf = float("inf")
            maneuver.maneuver = DubinsManeuver2D(
                [inf, inf, inf],
                [inf, inf, inf],
                inf,
                DubinsStruct(0.0, 0.0, 0.0, inf, "XXX"),
            )

    return maneuver


########## LSL ##########
def _LSL(maneuver, a, b, d, sa, ca, sb, cb):
    aux = np.arctan2(cb - ca, d + sa - sb)
    t = mod2pi(-a + aux)
    p = np.sqrt(2 + d**2 - 2 * np.cos(a - b) + 2 * d * (sa - sb))
    q = mod2pi(b - aux)
    length = (t + p + q) * maneuver.rhomin
    case = "LSL"
    return DubinsStruct(t, p, q, length, case)


########## RSR ##########
def _RSR(maneuver, a, b, d, sa, ca, sb, cb):
    aux = np.arctan2(ca - cb, d - sa + sb)
    t = mod2pi(a - aux)
    p = np.sqrt(2 + d**2 - 2 * np.cos(a - b) + 2 * d * (sb - sa))
    q = mod2pi(mod2pi(-b) + aux)
    length = (t + p + q) * maneuver.rhomin
    case = "RSR"
    return DubinsStruct(t, p, q, length, case)


########## LSR ##########
def _LSR(maneuver, a, b, d, sa, ca, sb, cb):
    aux1 = -2 + d**2 + 2 * np.cos(a - b) + 2 * d * (sa + sb)
    if aux1 > 0:
        p = np.sqrt(aux1)
        aux2 = math.atan2(-ca - cb, d + sa + sb) - math.atan(-2 / p)
        t = mod2pi(-a + aux2)
        q = mod2pi(-mod2pi(b) + aux2)
    else:
        t = p = q = float("inf")
    length = (t + p + q) * maneuver.rhomin
    case = "LSR"
    return DubinsStruct(t, p, q, length, case)


########## RSL ##########
def _RSL(maneuver, a, b, d, sa, ca, sb, cb):
    aux1 = d**2 - 2 + 2 * np.cos(a - b) - 2 * d * (sa + sb)
    if aux1 > 0:
        p = np.sqrt(aux1)
        aux2 = math.atan2(ca + cb, d - sa - sb) - math.atan(2 / p)
        t = mod2pi(a - aux2)
        q = mod2pi(mod2pi(b) - aux2)
    else:
        t = p = q = float("inf")
    length = (t + p + q) * maneuver.rhomin
    case = "RSL"
    return DubinsStruct(t, p, q, length, case)


########## RLR ##########
def _RLR(maneuver, a, b, d, sa, ca, sb, cb):
    aux = (6 - d**2 + 2 * np.cos(a - b) + 2 * d * (sa - sb)) / 8
    if abs(aux) <= 1:
        p = mod2pi(-math.acos(aux))
        t = mod2pi(a - math.atan2(ca - cb, d - sa + sb) + p / 2)
        q = mod2pi(a - b - t + p)
    else:
        t = p = q = float("inf")
    length = (t + p + q) * maneuver.rhomin
    case = "RLR"
    return DubinsStruct(t, p, q, length, case)


########## LRL ##########
def _LRL(maneuver, a, b, d, sa, ca, sb, cb):
    aux = (6 - d**2 + 2 * np.cos(a - b) + 2 * d * (-sa + sb)) / 8
    if abs(aux) <= 1:
        p = mod2pi(-math.acos(aux))
        t = mod2pi(-a + math.atan2(-ca + cb, d + sa - sb) + p / 2)
        q = mod2pi(b - a - t + p)
    else:
        t = p = q = float("inf")
    length = (t + p + q) * maneuver.rhomin
    case = "LRL"
    return DubinsStruct(t, p, q, length, case)


########## C ##########
def _C(maneuver):
    t = 0.0
    p = 2 * math.pi
    q = 0.0
    length = (t + p + q) * maneuver.rhomin
    case = "RRR"
    return DubinsStruct(t, p, q, length, case)


def getCoordinatesAt(maneuver, offset):
    # Offset normalizado
    noffset = offset / maneuver.rhomin

    # Translação para a origem
    qi = [0.0, 0.0, maneuver.qi[2]]

    # Gerando as configurações intermediárias
    l1 = maneuver.maneuver.t
    l2 = maneuver.maneuver.p
    q1 = getPositionInSegment(
        maneuver, l1, qi, maneuver.maneuver.case[0]
    )  # Final do segmento 1
    q2 = getPositionInSegment(
        maneuver, l2, q1, maneuver.maneuver.case[1]
    )  # Final do segmento 2

    # Obtendo o restante das configurações
    if noffset < l1:
        q = getPositionInSegment(maneuver, noffset, qi, maneuver.maneuver.case[0])
    elif noffset < (l1 + l2):
        q = getPositionInSegment(maneuver, noffset - l1, q1, maneuver.maneuver.case[1])
    else:
        q = getPositionInSegment(
            maneuver, noffset - l1 - l2, q2, maneuver.maneuver.case[2]
        )

    # Translação para a posição anterior
    q[0] = q[0] * maneuver.rhomin + maneuver.qi[0]
    q[1] = q[1] * maneuver.rhomin + maneuver.qi[1]
    q[2] = mod2pi(q[2])

    return q


def getPositionInSegment(maneuver, offset, qi, case):
    q = [0.0, 0.0, 0.0]
    if case == "L":
        q[0] = qi[0] + math.sin(qi[2] + offset) - math.sin(qi[2])
        q[1] = qi[1] - math.cos(qi[2] + offset) + math.cos(qi[2])
        q[2] = qi[2] + offset
    elif case == "R":
        q[0] = qi[0] - math.sin(qi[2] - offset) + math.sin(qi[2])
        q[1] = qi[1] + math.cos(qi[2] - offset) - math.cos(qi[2])
        q[2] = qi[2] - offset
    elif case == "S":
        q[0] = qi[0] + math.cos(qi[2]) * offset
        q[1] = qi[1] + math.sin(qi[2]) * offset
        q[2] = qi[2]
    return q


def getSamplingPoints(maneuver, res=0.1):
    points = []
    rng = np.arange(0.0, maneuver.maneuver.length, res)
    for offset in rng:
        points.append(getCoordinatesAt(maneuver, offset))
    return points


def DubinsManeuver3D_func(qi, qf, rhomin, pitchlims):
    maneuver = DubinsManeuver3D(qi, qf, rhomin, pitchlims, [], -1.0)

    # Delta Z (height)
    zi = maneuver.qi[2]
    zf = maneuver.qf[2]
    dz = zf - zi

    # Multiplication factor of rhomin in [1, 1000]
    a = 1.0
    b = 1.0

    fa = try_to_construct(maneuver, maneuver.rhomin * a)
    fb = try_to_construct(maneuver, maneuver.rhomin * b)

    while len(fb) < 2:
        b *= 2.0
        fb = try_to_construct(maneuver, maneuver.rhomin * b)

    if len(fa) > 0:
        maneuver.path = fa
    else:
        if len(fb) < 2:
            raise Exception("No maneuver exists")
    # Local optimalization between horizontal and vertical radii
    step = 0.1
    while abs(step) > 1e-10:
        c = b + step
        if c < 1.0:
            c = 1.0
        fc = try_to_construct(maneuver, maneuver.rhomin * c)
        if len(fc) > 0:
            if fc[1].maneuver.length < fb[1].maneuver.length:
                b = c
                fb = fc
                step *= 2.0
                continue
        step *= -0.1

    maneuver.path = fb
    Dlat, Dlon = fb
    maneuver.length = Dlon.maneuver.length
    return maneuver


def compute_sampling(maneuver, numberOfSamples=1000):
    Dlat, Dlon = maneuver.path
    # Sample points on the final path
    points = []
    lena = Dlon.maneuver.length
    rangeLon = lena * np.arange(numberOfSamples) / (numberOfSamples - 1)

    for ran in rangeLon:
        offsetLon = ran
        qSZ = getCoordinatesAt(Dlon, offsetLon)
        qXY = getCoordinatesAt(Dlat, qSZ[0])
        points.append([qXY[0], qXY[1], qSZ[1], qXY[2], qSZ[2]])

    return points


def try_to_construct(maneuver, horizontal_radius):
    qi2D = maneuver.qi[[0, 1, 3]]
    qf2D = maneuver.qf[[0, 1, 3]]

    Dlat = DubinsManeuver2D_func(qi2D, qf2D, rhomin=horizontal_radius)

    # After finding a long enough 2D curve, calculate the Dubins path on SZ axis
    qi3D = [0.0, maneuver.qi[2], maneuver.qi[4]]
    qf3D = [Dlat.maneuver.length, maneuver.qf[2], maneuver.qf[4]]

    vertical_curvature = np.sqrt(
        1.0 / maneuver.rhomin / maneuver.rhomin
        - 1.0 / horizontal_radius / horizontal_radius
    )
    if vertical_curvature < 1e-5:
        return []
    vertical_radius = 1.0 / vertical_curvature
    # Dlon = Vertical1D(qi3D, qf3D, vertical_radius, self.pitchlims)
    Dlon = DubinsManeuver2D_func(qi3D, qf3D, rhomin=vertical_radius)

    if Dlon.maneuver.case == "RLR" or Dlon.maneuver.case == "LRL":
        return []

    if Dlon.maneuver.case[0] == "R":
        if maneuver.qi[4] - Dlon.maneuver.t < maneuver.pitchlims[0]:
            return []
    else:
        if maneuver.qi[4] + Dlon.maneuver.t > maneuver.pitchlims[1]:
            return []

    # Final 3D path is formed by the two curves (Dlat, Dlon)
    return [Dlat, Dlon]


def getLowerBound(qi, qf, rhomin=1, pitchlims=[-np.pi / 4, np.pi / 2]):
    maneuver = DubinsManeuver3D(qi, qf, rhomin, pitchlims, [], -1.0)

    spiral_radius = rhomin * ((np.cos(max(-pitchlims[0], pitchlims[1]))) ** 2)

    qi2D = [maneuver.qi[i] for i in [0, 1, 3]]
    qf2D = [maneuver.qf[i] for i in [0, 1, 3]]
    Dlat = DubinsManeuver2D_func(qi2D, qf2D, rhomin=spiral_radius)

    qi3D = [0, maneuver.qi[2], maneuver.qi[4]]
    qf3D = [Dlat.maneuver.length, maneuver.qf[2], maneuver.qf[4]]

    Dlon = Vertical(qi3D, qf3D, maneuver.rhomin, maneuver.pitchlims)

    if Dlon.maneuver.case == "XXX":
        # TODO - update Vertical1D such that it compute the shortest prolongation
        maneuver.length = 0.0
        return maneuver

    maneuver.path = [Dlat, Dlon]
    maneuver.length = Dlon.maneuver.length
    return maneuver


def getUpperBound(qi, qf, rhomin=1, pitchlims=[-np.pi / 4, np.pi / 2]):
    maneuver = DubinsManeuver3D(qi, qf, rhomin, pitchlims, [], -1.0)

    safeRadius = np.sqrt(2) * maneuver.rhomin

    p1 = qi[0:2]
    p2 = qf[0:2]
    diff = p2 - p1
    dist = np.sqrt(diff[0] ** 2 + diff[1] ** 2)
    if dist < 4.0 * safeRadius:
        maneuver.length = np.inf
        return maneuver

    qi2D = [maneuver.qi[i] for i in [0, 1, 3]]
    qf2D = [maneuver.qf[i] for i in [0, 1, 3]]
    Dlat = DubinsManeuver2D(qi2D, qf2D, rhomin=safeRadius)

    qi3D = [0, maneuver.qi[1], maneuver.qi[3]]
    qf3D = [Dlat.maneuver.length, maneuver.qf[1], maneuver.qf[3]]

    Dlon = Vertical(qi3D, qf3D, safeRadius, maneuver.pitchlims)

    if Dlon.maneuver.case == "XXX":
        # TODO - update Vertical1D such that it compute the shortest prolongation
        maneuver.length = np.inf
        return maneuver

    maneuver.path = [Dlat, Dlon]
    maneuver.length = Dlon.maneuver.length
    return maneuver


import math
from .dubins_3d import DubinsStruct, DubinsManeuver2D_func, mod2pi

def Vertical(qi, qf, rhomin, pitchmax):
    maneuver = DubinsManeuver2D_func(
        qi, qf, rhomin, DubinsStruct(0.0, 0.0, 0.0, math.inf, "")
    )

    dx = maneuver.qf[0] - maneuver.qi[0]
    dy = maneuver.qf[1] - maneuver.qi[1]
    D = math.sqrt(dx**2 + dy**2)

    # Distance normalization
    d = D / maneuver.rhomin

    # Normalize the problem using rotation
    rotationAngle = mod2pi(math.atan2(dy, dx))
    # rotationAngle = mod2pi(math.atan2(dx, dy))
    a = mod2pi(maneuver.qi[2] - rotationAngle)
    b = mod2pi(maneuver.qf[2] - rotationAngle)

    # CSC
    pathLSL = v_LSL(maneuver)
    pathRSR = v_RSR(maneuver)
    pathLSR = v_LSR(maneuver, pitchmax)
    pathRSL = v_RSL(maneuver, pitchmax)
    _paths = [pathLSR, pathLSL, pathRSR, pathRSL]

    def a(x):
        return x.length

    _paths.sort(key=a)

    for p in _paths:
        # chech if the turns are too long (do not meet pitch constraint)
        if abs(p.t) < math.pi and abs(p.q) < math.pi:
            # check the inclination based on pitch constraint
            center_angle = maneuver.qi[2] + ((p.case[0] == "L") * p.t or -p.t)
            if center_angle < pitchmax[0] or center_angle > pitchmax[1]:
                continue
            maneuver.maneuver = p
            break

    if maneuver.maneuver is None:
        maneuver.maneuver = DubinsStruct(math.inf, math.inf, math.inf, math.inf, "XXX")

    return maneuver


########## LSL ##########
def v_LSL(maneuver):
    theta1 = maneuver.qi[2]
    theta2 = maneuver.qf[2]

    if theta1 <= theta2:
        # start/end points
        p1 = maneuver.qi[:2]
        p2 = maneuver.qf[:2]

        radius = maneuver.rhomin

        c1, s1 = radius * math.cos(theta1), radius * math.sin(theta1)
        c2, s2 = radius * math.cos(theta2), radius * math.sin(theta2)

        # origins of the turns
        o1 = p1 + [-s1, c1]
        o2 = p2 + [-s2, c2]

        diff = o2 - o1
        center_distance = math.sqrt(diff[0] ** 2 + diff[1] ** 2)
        centerAngle = math.atan2(diff[1], diff[0])
        # centerAngle = math.atan2(diff[0], diff[1])

        t = mod2pi(-theta1 + centerAngle)
        p = center_distance / radius
        q = mod2pi(theta2 - centerAngle)

        if t > math.pi:
            t = 0.0
            q = theta2 - theta1
            turn_end_y = o2[1] - radius * math.cos(theta1)
            diff_y = turn_end_y - p1[1]
            if abs(theta1) > 1e-5 and (diff_y < 0 == theta1 < 0):
                p = diff_y / math.sin(theta1) / radius
            else:
                t = p = q = float("inf")
        if q > math.pi:
            t = theta2 - theta1
            q = 0.0
            turn_end_y = o1[1] - radius * math.cos(theta2)
            diff_y = p2[1] - turn_end_y
            if abs(theta2) > 1e-5 and (diff_y < 0 == theta2 < 0):
                p = diff_y / math.sin(theta2) / radius
            else:
                t = p = q = float("inf")
    else:
        t = p = q = float("inf")

    length = (t + p + q) * maneuver.rhomin
    case = "LSL"

    return DubinsStruct(t, p, q, length, case)


def v_RSR(maneuver):
    theta1 = maneuver.qi[2]
    theta2 = maneuver.qf[2]

    if theta2 <= theta1:
        # start/end points
        p1 = maneuver.qi[0:2]
        p2 = maneuver.qf[0:2]

        radius = maneuver.rhomin

        c1, s1 = radius * math.cos(theta1), radius * math.sin(theta1)
        c2, s2 = radius * math.cos(theta2), radius * math.sin(theta2)

        # origins of the turns
        o1 = p1 + [s1, -c1]
        o2 = p2 + [s2, -c2]

        diff = o2 - o1
        center_distance = math.sqrt(diff[0] ** 2 + diff[1] ** 2)
        centerAngle = math.atan2(diff[1], diff[0])
        # centerAngle = math.atan2(diff[0], diff[1])

        t = maneuver._mod2pi(theta1 - centerAngle)
        p = center_distance / radius
        q = maneuver._mod2pi(-theta2 + centerAngle)

        if t > math.pi:
            t = 0.0
            q = -theta2 + theta1
            turn_end_y = o2[1] + radius * math.cos(theta1)
            diff_y = turn_end_y - p1[1]
            if abs(theta1) > 1e-5 and (diff_y < 0 == theta1 < 0):
                p = diff_y / math.sin(theta1) / radius
            else:
                t = p = q = math.inf
        if q > math.pi:
            t = -theta2 + theta1
            q = 0.0
            turn_end_y = o1[1] + radius * math.cos(theta2)
            diff_y = p2[1] - turn_end_y
            if abs(theta2) > 1e-5 and (diff_y < 0 == theta2 < 0):
                p = diff_y / math.sin(theta2) / radius
            else:
                t = p = q = math.inf
    else:
        t = p = q = math.inf

    length = (t + p + q) * maneuver.rhomin
    case = "RSR"

    return DubinsStruct(t, p, q, length, case)


def v_LSR(maneuver, pitchmax):
    theta1 = maneuver.qi[2]
    theta2 = maneuver.qf[2]

    # start/end points
    p1 = maneuver.qi[:2]
    p2 = maneuver.qf[:2]

    radius = maneuver.rhomin

    c1, s1 = radius * math.cos(theta1), radius * math.sin(theta1)
    c2, s2 = radius * math.cos(theta2), radius * math.sin(theta2)

    # origins of the turns
    o1 = p1 + [-s1, c1]
    o2 = p2 + [s2, -c2]

    diff = o2 - o1
    center_distance = math.sqrt(diff[0] ** 2 + diff[1] ** 2)

    # not constructible
    if center_distance < 2 * radius:
        diff[0] = math.sqrt(4.0 * radius * radius - diff[1] ** 2)
        alpha = math.pi / 2.0
    else:
        alpha = math.asin(2.0 * radius / center_distance)

    centerAngle = math.atan2(diff[1], diff[0]) + alpha
    # centerAngle = math.atan2(diff[0], diff[1]) + alpha

    if centerAngle < pitchmax[1]:
        t = mod2pi(-theta1 + centerAngle)
        p = math.sqrt(max(0.0, center_distance**2 - 4.0 * radius**2)) / radius
        q = mod2pi(-theta2 + centerAngle)
    else:
        centerAngle = pitchmax[1]
        t = mod2pi(-theta1 + centerAngle)
        q = mod2pi(-theta2 + centerAngle)

        # points on boundary between C and S segments
        c1, s1 = radius * math.cos(centerAngle), radius * math.sin(centerAngle)
        c2, s2 = radius * math.cos(centerAngle), radius * math.sin(centerAngle)
        w1 = o1 - [-s1, c1]
        w2 = o2 - [s2, -c2]

        p = (w2[1] - w1[1]) / math.sin(centerAngle) / radius

    length = (t + p + q) * maneuver.rhomin
    case = "LSR"

    return DubinsStruct(t, p, q, length, case)


def v_RSL(maneuver, pitchmax):
    theta1 = maneuver.qi[2]
    theta2 = maneuver.qf[2]

    # start/end points
    p1 = maneuver.qi[0:2]
    p2 = maneuver.qf[0:2]

    radius = maneuver.rhomin

    c1, s1 = radius * math.cos(theta1), radius * math.sin(theta1)
    c2, s2 = radius * math.cos(theta2), radius * math.sin(theta2)

    # origins of the turns
    o1 = p1 + [s1, -c1]
    o2 = p2 + [-s2, c2]

    diff = o2 - o1
    center_distance = math.sqrt(diff[0] ** 2 + diff[1] ** 2)

    # not constructible
    if center_distance < 2 * radius:
        diff[0] = math.sqrt(4.0 * radius * radius - diff[1] * diff[1])
        alpha = math.pi / 2.0
    else:
        alpha = math.arcsin(2.0 * radius / center_distance)

    centerAngle = math.arctan2(diff[1], diff[0]) - alpha

    if centerAngle > pitchmax[0]:
        t = mod2pi(theta1 - centerAngle)
        p = (
            math.sqrt(
                max(0.0, center_distance * center_distance - 4.0 * radius * radius)
            )
            / radius
        )
        q = mod2pi(theta2 - centerAngle)
    else:
        centerAngle = pitchmax[0]
        t = mod2pi(theta1 - centerAngle)
        q = mod2pi(theta2 - centerAngle)

        # points on boundary between C and S segments
        c1, s1 = radius * math.cos(centerAngle), radius * math.sin(centerAngle)
        c2, s2 = radius * math.cos(centerAngle), radius * math.sin(centerAngle)
        w1 = o1 - [s1, -c1]
        w2 = o2 - [-s2, c2]

        p = (w2[1] - w1[1]) / math.sin(centerAngle) / radius

    length = (t + p + q) * maneuver.rhomin
    case = "RSL"

    return DubinsStruct(t, p, q, length, case)
