from __future__ import annotations

import numpy as np

Vec3 = np.ndarray
Quat = np.ndarray


def quat(x: float = 0.0, y: float = 0.0, z: float = 0.0, w: float = 1.0) -> Quat:
    q = np.array([x, y, z, w], dtype=float)
    return normalize(q)


def normalize(q: Quat) -> Quat:
    q = np.asarray(q, dtype=float)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        raise ValueError("quaternion has zero length")
    return q / n


def multiply(a: Quat, b: Quat) -> Quat:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return normalize(
        np.array(
            [
                aw * bx + ax * bw + ay * bz - az * by,
                aw * by - ax * bz + ay * bw + az * bx,
                aw * bz + ax * by - ay * bx + az * bw,
                aw * bw - ax * bx - ay * by - az * bz,
            ],
            dtype=float,
        )
    )


def conjugate(q: Quat) -> Quat:
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=float)


def inverse(q: Quat) -> Quat:
    return conjugate(normalize(q))


def rotate(q: Quat, v: Vec3) -> Vec3:
    q = normalize(q)
    v = np.asarray(v, dtype=float)
    return multiply(multiply(q, np.array([v[0], v[1], v[2], 0.0])), conjugate(q))[:3]


def axis_angle(axis: Vec3, angle: float) -> Quat:
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    half = angle * 0.5
    return quat(*(axis * np.sin(half)), float(np.cos(half)))


def angle_between(a: Vec3, b: Vec3) -> tuple[Vec3, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    c = float(np.clip(np.dot(a, b), -1.0, 1.0))
    axis = np.cross(a, b)
    if np.linalg.norm(axis) < 1e-9:
        if c < 0.0:
            return np.array([1.0, 0.0, 0.0]), float(np.pi)
        return np.array([0.0, 0.0, 1.0]), 0.0
    return axis, float(np.arccos(c))


def rotation_between(a: Vec3, b: Vec3) -> Quat:
    axis, angle = angle_between(a, b)
    if abs(angle) < 1e-12:
        return quat()
    return axis_angle(axis, angle)


def euler_xyz(q: Quat) -> Vec3:
    x, y, z, w = normalize(q)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 0.999999:
        pitch = np.copysign(np.pi / 2.0, sinp)
        roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        yaw = 0.0
    else:
        roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        pitch = np.arcsin(sinp)
        yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([roll, pitch, yaw], dtype=float)


def vec3(value) -> Vec3:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (3,):
        raise ValueError("expected a 3D vector")
    return arr


def qlist(value) -> Quat:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (4,):
        raise ValueError("expected a quaternion [x, y, z, w]")
    return normalize(arr)
