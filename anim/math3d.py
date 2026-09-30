from __future__ import annotations

from math import atan2, asin, cos, sin, sqrt

Vec = tuple[float, float, float]
Quat = tuple[float, float, float, float]


def vadd(a: Vec, b: Vec) -> Vec:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def vsub(a: Vec, b: Vec) -> Vec:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def vscale(a: Vec, s: float) -> Vec:
    return (a[0] * s, a[1] * s, a[2] * s)


def vdot(a: Vec, b: Vec) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def vcross(a: Vec, b: Vec) -> Vec:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def vlength(a: Vec) -> float:
    return sqrt(vdot(a, a))


def vnorm(a: Vec) -> Vec:
    length = vlength(a)
    if length < 1e-12:
        raise ValueError("cannot normalize a zero-length vector")
    return vscale(a, 1.0 / length)


def vlerp(a: Vec, b: Vec, t: float) -> Vec:
    return tuple(a[i] + (b[i] - a[i]) * t for i in range(3))  # type: ignore[return-value]


def vclamp_scaled(a: Vec, max_length: float) -> Vec:
    length = vlength(a)
    if length <= max_length:
        return a
    return vscale(a, max_length / length)


def qidentity() -> Quat:
    return (1.0, 0.0, 0.0, 0.0)


def qnormalize(q: Quat) -> Quat:
    length = sqrt(sum(x * x for x in q))
    if length < 1e-12:
        raise ValueError("cannot normalize a zero-length quaternion")
    return tuple(x / length for x in q)  # type: ignore[return-value]


def qmul(a: Quat, b: Quat) -> Quat:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return qnormalize((
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + az * by - ay * bz,
        aw * by + ay * bw + ax * bz - az * bx,
        aw * bz - ax * by + ay * bx + az * bw,
    ))


def qmul_raw(a: Quat, b: Quat) -> Quat:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + az * by - ay * bz,
        aw * by + ay * bw + ax * bz - az * bx,
        aw * bz - ax * by + ay * bx + az * bw,
    )


def qconjugate(q: Quat) -> Quat:
    return (q[0], -q[1], -q[2], -q[3])


def qinverse(q: Quat) -> Quat:
    return qnormalize(qconjugate(q))


def q_rotate_vector(q: Quat, v: Vec) -> Vec:
    rotation = qnormalize(q)
    vector_q = (0.0, v[0], v[1], v[2])
    rotated = qmul_raw(qmul_raw(rotation, vector_q), qconjugate(rotation))
    return (rotated[1], rotated[2], rotated[3])


def q_shortest_arc(a: Vec, b: Vec) -> Quat:
    """Unit quaternion rotating unit vector a onto unit vector b along the shortest arc."""
    axis = vcross(a, b)
    dot = vdot(a, b)
    if vlength(axis) < 1e-10:
        return qidentity() if dot >= 0.0 else (-1.0, 0.0, 0.0, 0.0)
    q = (1.0 + dot, axis[0], axis[1], axis[2])
    return qnormalize(q)


def q_from_to_directions(current: Vec, desired: Vec) -> Quat:
    return q_shortest_arc(vnorm(current), vnorm(desired))


def quat_from_axes(x: Vec, y: Vec, z: Vec) -> Quat:
    x = vnorm(x)
    y = vnorm(y)
    z = vnorm(z)
    hand = vdot(vcross(x, y), z)
    if hand < 0:
        raise ValueError("bind axes must form a right-handed coordinate system")
    m00, m10, m20 = x
    m01, m11, m21 = y
    m02, m12, m22 = z
    trace = m00 + m11 + m22
    if trace > 0:
        s = sqrt(trace + 1.0) * 2.0
        return qnormalize((
            0.25 * s,
            (m12 - m21) / s,
            (m20 - m02) / s,
            (m01 - m10) / s,
        ))
    if m00 > m11 and m00 > m22:
        s = sqrt(1.0 + m00 - m11 - m22) * 2
        return qnormalize((
            (m12 - m21) / s,
            0.25 * s,
            (m01 + m10) / s,
            (m20 + m02) / s,
        ))
    if m11 > m22:
        s = sqrt(1.0 + m11 - m00 - m22) * 2
        return qnormalize((
            (m20 - m02) / s,
            (m01 + m10) / s,
            0.25 * s,
            (m12 + m21) / s,
        ))
    s = sqrt(1.0 + m22 - m00 - m11) * 2
    return qnormalize((
        (m10 - m01) / s,
        (m20 + m02) / s,
        (m12 + m21) / s,
        0.25 * s,
    ))


def quat_to_euler_xyz(q: Quat) -> Vec:
    w, x, y, z = qnormalize(q)
    sin_pitch = 2.0 * (w * y - z * x)
    sin_pitch = max(-1.0, min(1.0, sin_pitch))
    pitch = asin(sin_pitch)
    yaw = atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    roll = atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    return (roll, pitch, yaw)


def quat_from_euler_xyz(euler: Vec) -> Quat:
    roll, pitch, yaw = euler
    cr, sr = cos(roll * 0.5), sin(roll * 0.5)
    cp, sp = cos(pitch * 0.5), sin(pitch * 0.5)
    cy, sy = cos(yaw * 0.5), sin(yaw * 0.5)
    return qnormalize((
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ))


def almost_zero(v: Vec, eps: float = 1e-8) -> bool:
    return all(abs(x) <= eps for x in v)
