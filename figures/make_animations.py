"""Generate the kinematics animations used in the README.

Usage:  python figures/make_animations.py [name ...]
Writes <name>.mp4 and <name>.gif next to this file. Requires numpy, matplotlib, ffmpeg.
"""
import os
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Circle, Ellipse, Polygon

OUT = os.path.dirname(os.path.abspath(__file__))
FPS, N = 20, 120

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e6e5e1"
BLUE, ORANGE, AQUA, BLUE_LIGHT = "#2a78d6", "#eb6834", "#1baf7a", "#9ec5f4"
LINK = "#3d3c3a"

if any("Inter" == f.name for f in font_manager.fontManager.ttflist):
    plt.rcParams["font.family"] = "Inter"
plt.rcParams.update({"font.size": 11, "text.color": INK, "axes.edgecolor": GRID})


def canvas(ax, title, subtitle, xlim, ylim):
    ax.cla()
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_facecolor(SURFACE)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.figure.texts.clear()
    ax.figure.text(0.05, 0.93, title, fontsize=15, fontweight="bold", color=INK, va="center")
    ax.figure.text(0.05, 0.875, subtitle, fontsize=10.5, color=INK2, va="center")


def ground(ax, x, y, w=0.5):
    ax.plot([x - w / 2, x + w / 2], [y, y], color=MUTED, lw=2, solid_capstyle="round", zorder=1)
    for k in np.linspace(x - w / 2, x + w / 2, 6)[:-1]:
        ax.plot([k, k + w / 12], [y - 0.07, y], color=MUTED, lw=1, zorder=1)


def arm(ax, pts, color=LINK, alpha=1.0, lw=7, joints=True, z=5):
    pts = np.asarray(pts)
    ax.plot(pts[:, 0], pts[:, 1], color=color, lw=lw, alpha=alpha, solid_capstyle="round",
            solid_joinstyle="round", zorder=z)
    if joints:
        ax.scatter(pts[:-1, 0], pts[:-1, 1], s=70, color=SURFACE, edgecolor=color, linewidth=2,
                   alpha=alpha, zorder=z + 1)


def tip(ax, p, color=BLUE, z=9):
    ax.scatter([p[0]], [p[1]], s=90, color=color, edgecolor=SURFACE, linewidth=2, zorder=z)


def fk(L, q):
    a = np.cumsum(q)
    return np.vstack([[0, 0], np.cumsum(np.c_[L * np.cos(a), L * np.sin(a)], axis=0)])


def ik2(L1, L2, p, elbow=1.0):
    d2 = p[0] ** 2 + p[1] ** 2
    c = np.clip((d2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2), -1, 1)
    q2 = elbow * np.arccos(c)
    q1 = np.arctan2(p[1], p[0]) - np.arctan2(L2 * np.sin(q2), L1 + L2 * np.cos(q2))
    return np.array([q1, q2])


def note(ax, lines, x=0.05, y=0.80):
    for i, (txt, col) in enumerate(lines):
        ax.figure.text(x, y - 0.045 * i, txt, fontsize=10.5, color=col, va="center",
                       family="DejaVu Sans Mono")


# 1 ------------------------------------------------------------------ forward kinematics
def forward_kinematics(ax, i, st):
    L = np.array([1.0, 0.8, 0.5])
    t = 2 * np.pi * i / N
    q = np.array([0.9 + 0.6 * np.sin(t), -0.9 + 0.8 * np.sin(t + 1.3), 0.6 * np.sin(2 * t + 0.4)])
    P = fk(L, q)
    tr = np.array([fk(L, np.array([0.9 + 0.6 * np.sin(s), -0.9 + 0.8 * np.sin(s + 1.3),
                                   0.6 * np.sin(2 * s + 0.4)]))[-1]
                   for s in 2 * np.pi * np.arange(i - 45, i + 1) / N])
    canvas(ax, "Forward kinematics", "Joint angles in, end-effector pose out", (-1.7, 3.58), (-0.95, 2.5))
    ground(ax, 0, -0.12)
    ax.plot(tr[:, 0], tr[:, 1], color=BLUE_LIGHT, lw=2.5, solid_capstyle="round", zorder=2)
    arm(ax, P)
    tip(ax, P[-1])
    d = np.degrees(q)
    note(ax, [(f"θ1 = {d[0]:6.1f}°", INK2), (f"θ2 = {d[1]:6.1f}°", INK2), (f"θ3 = {d[2]:6.1f}°", INK2),
              (f"x  = {P[-1,0]:6.2f}", BLUE), (f"y  = {P[-1,1]:6.2f}", BLUE)])


# 2 ------------------------------------------------------ inverse kinematics, two solutions
def inverse_kinematics(ax, i, st):
    t = 2 * np.pi * i / N
    p = np.array([1.05 + 0.55 * np.cos(t), 0.55 + 0.45 * np.sin(2 * t)])
    path = np.array([[1.05 + 0.55 * np.cos(s), 0.55 + 0.45 * np.sin(2 * s)] for s in np.linspace(0, 2 * np.pi, 200)])
    canvas(ax, "Inverse kinematics", "One target pose, two joint solutions for a 2R arm", (-1.6, 2.6), (-1.05, 2.1))
    ax.add_patch(Circle((0, 0), 2.0, fill=False, ls=(0, (4, 4)), ec=GRID, lw=1.5, zorder=0))
    ground(ax, 0, -0.12)
    ax.plot(path[:, 0], path[:, 1], color=GRID, lw=2, zorder=1)
    arm(ax, fk(np.array([1.0, 1.0]), ik2(1, 1, p, -1)), color=ORANGE, z=4)
    arm(ax, fk(np.array([1.0, 1.0]), ik2(1, 1, p, +1)), color=BLUE, z=6)
    tip(ax, p, color=INK)
    ax.plot([], [], color=BLUE, lw=5, label="Elbow down")
    ax.plot([], [], color=ORANGE, lw=5, label="Elbow up")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.93), frameon=False, labelcolor=INK2, handlelength=1.4)
    ax.text(1.42, 1.48, "workspace boundary", color=MUTED, fontsize=9, rotation=-45, ha="center")


# 3 ------------------------------------------------------------------ numerical IK
def numerical_ik(ax, i, st):
    L = np.array([1.0, 0.8, 0.6])
    targets = [np.array([1.3, 1.2]), np.array([-0.4, 1.7]), np.array([1.9, 0.2])]
    seg = N // 3
    if "q" not in st:
        st["q"] = np.array([0.3, 0.5, 0.4])
        for k in range(N):  # warm-up cycle so the loop closes
            _step(L, st, targets[(k // seg) % 3])
        st["hist"] = []
    k, tgt = i // seg, targets[(i // seg) % 3]
    if i % seg == 0:
        st["hist"] = []
        st["it"] = 0
    if i % seg >= 6:  # hold a few frames on the new target before iterating
        if len(st["hist"]) < 40 and st["it"] % 3 == 0:
            st["hist"].append(fk(L, st["q"]))
        _step(L, st, tgt)
        st["it"] += 1
    P = fk(L, st["q"])
    err = np.linalg.norm(tgt - P[-1])
    canvas(ax, "Numerical inverse kinematics", "Damped least squares: iterate along the Jacobian toward the target",
           (-1.9, 2.9), (-0.75, 2.85))
    ground(ax, 0, -0.12)
    for g in st["hist"]:
        arm(ax, g, color=BLUE_LIGHT, alpha=0.45, lw=3, joints=False, z=2)
    arm(ax, P)
    tip(ax, P[-1])
    ax.scatter([tgt[0]], [tgt[1]], s=170, marker="X", color=ORANGE, edgecolor=SURFACE, linewidth=1.5, zorder=10)
    ax.annotate("target", tgt, xytext=(10, 10), textcoords="offset points", color=INK2, fontsize=10)
    note(ax, [(f"iteration   {st.get('it', 0):3d}", INK2), (f"error     {err:5.3f}", BLUE)])


def _step(L, st, tgt, lam=0.25, alpha=0.18):
    q = st["q"]
    a = np.cumsum(q)
    P = fk(L, q)
    J = np.zeros((2, 3))
    for j in range(3):
        r = P[-1] - P[j]
        J[:, j] = [-r[1], r[0]]
    e = tgt - P[-1]
    st["q"] = q + alpha * J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(2), e)


# 4 ------------------------------------------------------------------ null-space motion
def null_space_motion(ax, i, st):
    L = np.array([1.0, 0.9, 0.6])
    p = np.array([1.35, 0.85])
    t = 2 * np.pi * i / N

    def cfg(s):
        phi = -0.35 + 1.05 * np.sin(s)
        w = p - L[2] * np.array([np.cos(phi), np.sin(phi)])
        q12 = ik2(L[0], L[1], w, -1.0)
        return np.array([q12[0], q12[1], phi - q12.sum()])

    canvas(ax, "Redundancy and null-space motion", "A 3R arm moves its joints while the end-effector stays put",
           (-1.4, 2.8), (-0.45, 2.3))
    ground(ax, 0, -0.12)
    for s in np.linspace(-np.pi / 2, np.pi / 2, 7):
        arm(ax, fk(L, cfg(s)), color=BLUE_LIGHT, alpha=0.35, lw=3, joints=False, z=2)
    q = cfg(t)
    arm(ax, fk(L, q))
    tip(ax, p, color=ORANGE)
    ax.annotate("fixed end-effector\nposition", p, xytext=(16, 4), textcoords="offset points", color=INK2,
                fontsize=10, va="center")
    d = np.degrees(q)
    note(ax, [(f"θ1 = {d[0]:6.1f}°", INK2), (f"θ2 = {d[1]:6.1f}°", INK2), (f"θ3 = {d[2]:6.1f}°", INK2)])


# 5 ------------------------------------------------------------------ manipulability
def manipulability(ax, i, st):
    L1 = L2 = 1.0
    t = 2 * np.pi * i / N
    q = np.array([0.95 + 0.25 * np.sin(t), -(0.08 + (np.pi - 0.16) * 0.5 * (1 - np.cos(t)))])
    P = fk(np.array([L1, L2]), q)
    s1, s12, c1, c12 = np.sin(q[0]), np.sin(q.sum()), np.cos(q[0]), np.cos(q.sum())
    J = np.array([[-L1 * s1 - L2 * s12, -L2 * s12], [L1 * c1 + L2 * c12, L2 * c12]])
    U, S, _ = np.linalg.svd(J)
    w = abs(np.linalg.det(J))
    canvas(ax, "Manipulability ellipse", "How easily the end-effector can move in each direction",
           (-1.95, 3.4), (-1.25, 2.25))
    ground(ax, 0, -0.12)
    ang = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
    k = 0.3
    ax.add_patch(Ellipse(P[-1], 2 * k * S[0], 2 * k * max(S[1], 0.012), angle=ang, fc=BLUE, alpha=0.18, ec="none", zorder=3))
    ax.add_patch(Ellipse(P[-1], 2 * k * S[0], 2 * k * max(S[1], 0.012), angle=ang, fill=False, ec=BLUE, lw=2, zorder=7))
    arm(ax, P)
    tip(ax, P[-1])
    lines = [(f"w = |det J| = {w:4.2f}", BLUE)]
    if w < 0.25:
        lines.append(("near a singularity:", INK2))
        lines.append(("one direction is lost", INK2))
    note(ax, lines)


# 6 ------------------------------------------------------------------ mobile robot
def _mobile_sim():
    s = np.linspace(0, 2 * np.pi, 600, endpoint=False)
    path = np.c_[2.4 * np.sin(s), 1.25 * np.sin(2 * s)]
    Lw, v, Ld, dt = 0.55, 1.0, 0.75, 0.02
    x = np.array([path[0, 0], path[0, 1], np.arctan2(path[1, 1] - path[0, 1], path[1, 0] - path[0, 0])])
    seg = np.linalg.norm(np.diff(np.vstack([path, path[:1]]), axis=0), axis=1)
    total = seg.sum()
    steps = int(round(2 * total / (v * dt)))
    out, idx = [], 0
    for _ in range(steps):
        d = np.linalg.norm(path - x[:2], axis=1)
        win = [(idx + j) % len(path) for j in range(60)]
        idx = win[int(np.argmin(d[win]))]
        j = idx
        while np.linalg.norm(path[j] - x[:2]) < Ld:
            j = (j + 1) % len(path)
        g = path[j]
        a = np.arctan2(g[1] - x[1], g[0] - x[0]) - x[2]
        delta = np.arctan2(2 * Lw * np.sin(a), Ld)
        out.append((x.copy(), delta, g.copy()))
        x = x + dt * np.array([v * np.cos(x[2]), v * np.sin(x[2]), v * np.tan(delta) / Lw])
    out = out[len(out) // 2:]  # second lap only, so the loop closes
    pick = np.linspace(0, len(out), N, endpoint=False).astype(int)
    return path, [out[k] for k in pick], Lw


def mobile_robot(ax, i, st):
    if "sim" not in st:
        st["sim"] = _mobile_sim()
    path, fr, Lw = st["sim"]
    x, delta, g = fr[i]
    canvas(ax, "Mobile robot kinematics", "Kinematic bicycle model tracking a path with pure pursuit",
           (-3.3, 3.3), (-2.3, 2.65))
    ax.plot(*np.vstack([path, path[:1]]).T, color=GRID, lw=6, solid_capstyle="round", zorder=1)
    tr = np.array([fr[(i - k) % N][0][:2] for k in range(28, -1, -1)])
    ax.plot(tr[:, 0], tr[:, 1], color=BLUE_LIGHT, lw=3, solid_capstyle="round", zorder=2)
    c, s = np.cos(x[2]), np.sin(x[2])
    R = np.array([[c, -s], [s, c]])
    body = np.array([[-0.18, -0.2], [Lw + 0.18, -0.2], [Lw + 0.18, 0.2], [-0.18, 0.2]]) @ R.T + x[:2]
    ax.add_patch(Polygon(body, closed=True, fc=BLUE, ec=SURFACE, lw=2, alpha=0.9, zorder=5, joinstyle="round"))
    for wx, ang in [(0.0, 0.0), (Lw, delta)]:
        for wy in (-0.24, 0.24):
            cw = np.array([wx, wy]) @ R.T + x[:2]
            d = np.array([np.cos(x[2] + ang), np.sin(x[2] + ang)]) * 0.13
            ax.plot([cw[0] - d[0], cw[0] + d[0]], [cw[1] - d[1], cw[1] + d[1]], color=LINK, lw=5,
                    solid_capstyle="round", zorder=6)
    front = np.array([Lw, 0]) @ R.T + x[:2]
    ax.plot([x[0], g[0]], [x[1], g[1]], color=ORANGE, lw=1.5, ls=(0, (3, 3)), zorder=4)
    ax.scatter([g[0]], [g[1]], s=80, color=ORANGE, edgecolor=SURFACE, linewidth=2, zorder=7)
    ax.annotate("look-ahead point", g, xytext=(9, 9), textcoords="offset points", color=INK2, fontsize=10)
    note(ax, [(f"heading   {np.degrees((x[2] + np.pi) % (2 * np.pi) - np.pi):6.1f}°", INK2),
              (f"steering  {np.degrees(delta):6.1f}°", BLUE)], x=0.70, y=0.20)


# 7 ------------------------------------------------------------------ continuum robot
def _cc_section(p0, th0, kappa, length, n=30):
    s = np.linspace(0, length, n)
    if abs(kappa) < 1e-6:
        x, y = s, np.zeros_like(s)
    else:
        x, y = np.sin(kappa * s) / kappa, (1 - np.cos(kappa * s)) / kappa
    c, sn = np.cos(th0), np.sin(th0)
    pts = np.c_[c * x - sn * y, sn * x + c * y] + p0
    return pts, th0 + kappa * s


def continuum(ax, i, st):
    t = 2 * np.pi * i / N

    def shape(s):
        k1, k2 = 1.1 * np.sin(s), 1.9 * np.sin(2 * s + 0.6)
        a, tha = _cc_section(np.zeros(2), np.pi / 2, k1, 1.1)
        b, thb = _cc_section(a[-1], tha[-1], k2, 1.0)
        return a, tha, b, thb, k1, k2

    a, tha, b, thb, k1, k2 = shape(t)
    tr = np.array([shape(s)[2][-1] for s in 2 * np.pi * np.arange(i - 40, i + 1) / N])
    canvas(ax, "Continuum robot kinematics", "Two constant-curvature sections: curvature in, backbone shape out",
           (-2.3, 2.3), (-0.45, 2.75))
    ground(ax, 0, -0.03, w=0.7)
    ax.plot(tr[:, 0], tr[:, 1], color=GRID, lw=2.5, solid_capstyle="round", zorder=1)
    for pts, th, col in [(a, tha, BLUE), (b, thb, ORANGE)]:
        ax.plot(pts[:, 0], pts[:, 1], color=col, lw=9, solid_capstyle="round", zorder=4)
        for j in range(0, len(pts), 5):
            n = np.array([-np.sin(th[j]), np.cos(th[j])]) * 0.11
            ax.plot([pts[j, 0] - n[0], pts[j, 0] + n[0]], [pts[j, 1] - n[1], pts[j, 1] + n[1]], color=LINK,
                    lw=2.5, solid_capstyle="round", zorder=5)
    tip(ax, b[-1], color=INK)
    ax.plot([], [], color=BLUE, lw=6, label=f"Section 1   κ = {k1:5.2f}")
    ax.plot([], [], color=ORANGE, lw=6, label=f"Section 2   κ = {k2:5.2f}")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.93), frameon=False, labelcolor=INK2, handlelength=1.4,
              prop={"family": "DejaVu Sans Mono", "size": 10})


# 8 ------------------------------------------------------------------ parallel five-bar
def five_bar(ax, i, st):
    A1, A2, l1, l2 = np.array([-0.5, 0.0]), np.array([0.5, 0.0]), 1.0, 1.25
    t = 2 * np.pi * i / N
    path = np.array([[0.55 * np.cos(s), 1.45 + 0.35 * np.sin(s)] for s in np.linspace(0, 2 * np.pi, 200)])
    p = np.array([0.55 * np.cos(t), 1.45 + 0.35 * np.sin(t)])
    qa = ik2(l1, l2, p - A1, -1.0)
    qb = ik2(l1, l2, p - A2, +1.0)
    B1 = A1 + l1 * np.array([np.cos(qa[0]), np.sin(qa[0])])
    B2 = A2 + l1 * np.array([np.cos(qb[0]), np.sin(qb[0])])
    canvas(ax, "Parallel kinematics", "A five-bar mechanism: two actuated joints share one end-effector",
           (-2.2, 2.2), (-0.5, 2.55))
    ground(ax, A1[0], -0.12, w=0.4)
    ground(ax, A2[0], -0.12, w=0.4)
    ax.plot(path[:, 0], path[:, 1], color=GRID, lw=2, zorder=1)
    arm(ax, [A1, B1, p], color=BLUE)
    arm(ax, [A2, B2, p], color=ORANGE)
    tip(ax, p, color=INK)
    note(ax, [(f"left  actuator {np.degrees(qa[0]):6.1f}°", BLUE), (f"right actuator {np.degrees(qb[0]):6.1f}°", ORANGE)],
         x=0.05, y=0.80)


# 9 ------------------------------------------------------------------ Jacobian matrix
def _arrow(ax, p, v, color, lw=2.5, z=8):
    ax.annotate("", xy=(p[0] + v[0], p[1] + v[1]), xytext=(p[0], p[1]), zorder=z,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=16, shrinkA=0, shrinkB=0))


def jacobian(ax, i, st):
    L = np.array([1.0, 0.85])
    t = 2 * np.pi * i / N
    w = 2 * np.pi / (N / FPS)
    q = np.array([0.75 + 0.45 * np.sin(t), 1.25 + 0.65 * np.sin(2 * t + 1.0)])
    qd = np.array([0.45 * np.cos(t), 1.3 * np.cos(2 * t + 1.0)]) * w
    P = fk(L, q)
    J = np.array([[-(P[2, 1] - P[j, 1]), P[2, 0] - P[j, 0]] for j in range(2)]).T
    v = J @ qd
    k = 0.45
    canvas(ax, "The Jacobian matrix", "Each column is the end-effector velocity produced by one joint turning alone",
           (-1.75, 3.05), (-0.6, 2.55))
    ground(ax, 0, -0.12)
    for j, col in enumerate((BLUE, ORANGE)):
        ax.plot([P[j, 0], P[2, 0]], [P[j, 1], P[2, 1]], color=col, lw=1.2, ls=(0, (3, 3)), alpha=0.7, zorder=2)
    arm(ax, P)
    ax.scatter(P[:2, 0], P[:2, 1], s=70, color=SURFACE, edgecolor=[BLUE, ORANGE], linewidth=2.5, zorder=7)
    _arrow(ax, P[2], k * J[:, 0], BLUE)
    _arrow(ax, P[2], k * J[:, 1], ORANGE)
    _arrow(ax, P[2], k * 1.6 * v, INK, lw=3, z=9)
    tip(ax, P[2], color=INK, z=10)
    ax.plot([], [], color=BLUE, lw=3, label="Column 1: joint 1 alone")
    ax.plot([], [], color=ORANGE, lw=3, label="Column 2: joint 2 alone")
    ax.plot([], [], color=INK, lw=3, label="Velocity  v = J q̇")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.0), frameon=False, labelcolor=INK2, handlelength=1.4)
    note(ax, [(f"J = [ {J[0,0]:5.2f}  {J[0,1]:5.2f} ]", INK2), (f"    [ {J[1,0]:5.2f}  {J[1,1]:5.2f} ]", INK2),
              (f"det J = {np.linalg.det(J):5.2f}", INK2)])


# ================================================================== worked examples
def _stage(i, durs):
    """Return (stage index, progress 0..1 within that stage) for frame i."""
    acc = 0
    for k, dur in enumerate(durs):
        if i < acc + dur:
            u = (i - acc + 1) / dur
            return k, u * u * (3 - 2 * u)
        acc += dur
    return len(durs) - 1, 1.0


def _steps(ax, lines, y=0.80):
    for k, (txt, col) in enumerate(lines):
        ax.figure.text(0.05, y - 0.047 * k, txt, fontsize=10.5, color=col, va="center", family="DejaVu Sans Mono")


def _arc(ax, c, a0, a1, r, color):
    a = np.linspace(a0, a1, 40)
    ax.plot(c[0] + r * np.cos(a), c[1] + r * np.sin(a), color=color, lw=2, zorder=4)


# 10 ---------------------------------------------- solved problem: forward kinematics
def _triad(ax, T, s=0.28):
    o = T[:2, 3]
    for c, col in enumerate((RED, AQUA)):
        e = o + s * T[:2, c]
        ax.plot([o[0], e[0]], [o[1], e[1]], color=col, lw=2.2, solid_capstyle="round", zorder=8)


def example_fk(ax, i, st):
    L = np.array([1.0, 0.8, 0.5])
    goal = np.radians([30.0, 45.0, -30.0])
    k, u = _stage(i, [30, 30, 20, 30, 20, 30, 90])
    prog = [0, 0, 0]
    for j, move in enumerate((1, 3, 5)):
        prog[j] = u if k == move else float(k > move)
    q = goal * np.array(prog)
    P = fk(L, q)
    a = np.cumsum(q)
    A = [dh(goal[j], 0.0, L[j], 0.0) for j in range(3)]
    canvas(ax, "Solved problem: forward kinematics", "Planar 3R arm from a DH table. Where is the end-effector?",
           (-3.15, 2.35), (-0.6, 3.0))
    ground(ax, 0, -0.12)
    cols = (BLUE, ORANGE, AQUA)
    for j in range(3):
        if prog[j] > 0:
            a0 = a[j] - q[j]
            ref = P[j] + 0.55 * np.array([np.cos(a0), np.sin(a0)])
            ax.plot([P[j, 0], ref[0]], [P[j, 1], ref[1]], color=MUTED, lw=1.2, ls=(0, (3, 3)), zorder=2)
            _arc(ax, P[j], a0, a[j], 0.32, cols[j])
    arm(ax, P)
    Tc = np.eye(4)
    _triad(ax, Tc)
    for j in range(3):  # frame j+1 appears once its transform has been applied
        Tc = Tc @ dh(q[j], 0.0, L[j], 0.0)
        if prog[j] >= 1.0:
            _triad(ax, Tc)
    tip(ax, P[-1], color=INK)
    if k == 0:
        lines = [("DH table (m)", INK), (" i    θ     d    a    α", INK2), (" 1   30°    0   1.0   0", INK2),
                 (" 2   45°    0   0.8   0", INK2), (" 3  −30°    0   0.5   0", INK2), ("", INK2),
                 ("One matrix A per row.", INK2), ("Frame axes: x red, y green", INK2)]
    elif k <= 5:
        n = (k + 1) // 2
        lines = [(f"1  Link transform {n} of 3", INK), (f"   frame {n - 1} → frame {n}", INK2), ("", INK2)]
        lines += [(s, INK2) for s in _mat(f"A{n} =", A[n - 1])]
    else:
        m = _mat("2  T = A1 A2 A3 =", A[0] @ A[1] @ A[2])
        lines = [(m[0], INK)] + [(s, INK2) for s in m[1:]]
        lines += [("", INK2), ("Answer", INK), ("last column:  x = 1.43 m", INK), ("              y = 1.63 m", INK),
                  ("rotation block: 45°", INK)]
        ax.annotate("(1.43, 1.63)", P[-1], xytext=(14, -16), textcoords="offset points", color=INK2, fontsize=10)
    for n_, (txt, col) in enumerate(lines):
        ax.figure.text(0.05, 0.80 - 0.046 * n_, txt, fontsize=9.5, color=col, va="center", family="DejaVu Sans Mono")


example_fk.frames = 250


# 11 ---------------------------------------------- solved problem: inverse kinematics
def example_ik(ax, i, st):
    l1, l2, p = 1.0, 0.8, np.array([1.2, 0.9])
    k, u = _stage(i, [35, 30, 30, 20, 30, 30, 85])
    canvas(ax, "Solved problem: inverse kinematics", "Planar 2R arm from a DH table. Which joint angles reach (1.2, 0.9)?",
           (-3.45, 2.05), (-1.15, 2.45))
    ground(ax, 0, -0.12)
    ax.scatter([p[0]], [p[1]], s=170, marker="X", color=INK, edgecolor=SURFACE, linewidth=1.5, zorder=10)
    ax.annotate("target", p, xytext=(10, 10), textcoords="offset points", color=INK2, fontsize=10)
    lines = [(" i    θ    d    a    α", INK2), (" 1   θ1    0   1.0   0", INK2), (" 2   θ2    0   0.8   0", INK2),
             ("Last column of A1 A2:", INK), ("  x = c1 + 0.8 c12 = 1.2", INK2), ("  y = s1 + 0.8 s12 = 0.9", INK2), ("", INK2)]
    if k >= 1:
        f = u if k == 1 else 1.0
        ax.plot([0, f * p[0]], [0, f * p[1]], color=MUTED, lw=1.5, ls=(0, (3, 3)), zorder=2)
        lines += [("1  x² + y² = r² = 2.25", INK)]
    if k >= 2:
        f = u if k == 2 else 1.0
        a = np.linspace(0, 2 * np.pi * f, 120)
        ax.plot(l1 * np.cos(a), l1 * np.sin(a), color=GRID, lw=1.8, zorder=1)
        ax.plot(p[0] + l2 * np.cos(a + np.pi), p[1] + l2 * np.sin(a + np.pi), color=GRID, lw=1.8, zorder=1)
        lines += [("2  cos θ2 = 0.381", INK), ("   θ2 = ±67.6°", INK)]
    qa, qb = ik2(l1, l2, p, +1), ik2(l1, l2, p, -1)
    if k >= 3:
        for q in (qa, qb):
            e = l1 * np.array([np.cos(q[0]), np.sin(q[0])])
            ax.scatter([e[0]], [e[1]], s=60, color=SURFACE, edgecolor=MUTED, linewidth=2, zorder=3)
        lines += [("3  θ1 = 36.9° ∓ 29.5°", INK)]
    if k >= 4:
        arm(ax, fk(np.array([l1, l2]), qa), color=BLUE, alpha=u if k == 4 else 1.0, z=6)
        lines += [("", INK2), ("Elbow down", BLUE), ("θ1 =  7.3°   θ2 =  67.6°", BLUE)]
    if k >= 5:
        arm(ax, fk(np.array([l1, l2]), qb), color=ORANGE, alpha=u if k == 5 else 1.0, z=5)
        lines += [("Elbow up", ORANGE), ("θ1 = 66.4°   θ2 = −67.6°", ORANGE)]
    for n_, (txt, col) in enumerate(lines):
        ax.figure.text(0.05, 0.80 - 0.046 * n_, txt, fontsize=9.5, color=col, va="center", family="DejaVu Sans Mono")


example_ik.frames = 260


# 12 ---------------------------------------------- solved problem: Jacobian
def example_jacobian(ax, i, st):
    L = np.array([1.0, 0.8])
    q0, qd = np.radians([30.0, 60.0]), np.array([0.5, -1.0])
    k, u = _stage(i, [35, 30, 20, 30, 20, 40, 30, 65])
    canvas(ax, "Solved problem: the Jacobian", "Planar 2R arm from a DH table, joint rates (0.5, −1.0) rad/s. How fast is the tip?",
           (-3.35, 2.15), (-0.6, 3.0))
    ground(ax, 0, -0.12)
    q = q0 + qd * 0.35 * u if k == 6 else (q0 + qd * 0.35 if k == 7 else q0)
    P0, P = fk(L, q0), fk(L, q)
    J = np.array([[-1.3, -0.8], [0.866, 0.0]])
    v = J @ qd
    for j, (stg, col) in enumerate([(1, BLUE), (3, ORANGE)]):  # one joint swings alone
        if k == stg:
            dq = np.zeros(2)
            dq[j] = 0.35 * np.sin(np.pi * u)
            arm(ax, fk(L, q0 + dq), color=col, alpha=0.35, lw=4, joints=False, z=3)
    if k >= 6:
        arm(ax, P0, color=MUTED, alpha=0.35, lw=4, joints=False, z=2)
    arm(ax, P)
    e = P0[2]
    if k <= 4:
        for name, pt in (("p0", P0[0]), ("p1", P0[1]), ("p2", P0[2])):
            ax.annotate(name, pt, xytext=(-24, 4) if name == "p0" else (9, -13), textcoords="offset points",
                        color=INK2, fontsize=10)
    faded = k >= 5
    if k == 0:
        lines = [("DH table (m)", INK), (" i    θ     d    a    α", INK2), (" 1   30°    0   1.0   0", INK2),
                 (" 2   60°    0   0.8   0", INK2), ("", INK2), ("A1 and A1 A2 give each", INK2), ("frame's origin p and axis z.", INK2)]
    elif k <= 4:
        lines = [("1  From A1 and A1 A2:", INK), ("   p1 = (0.87, 0.50)", INK2), ("   p2 = (0.87, 1.30)", INK2),
                 ("   z0 = z1 = (0, 0, 1)", INK2), ("", INK2), ("2  Column i = z × (p2 − p)", INK),
                 ("   J1 = z0 × (p2 − p0)", BLUE), ("      = (−1.30, 0.87)", BLUE)]
        if k >= 3:
            lines += [("   J2 = z1 × (p2 − p1)", ORANGE), ("      = (−0.80, 0.00)", ORANGE)]
    else:
        lines = [("2  J = [ −1.30  −0.80 ]", INK), ("       [  0.87   0.00 ]", INK), ("", INK2),
                 ("3  v = J q̇ = 0.5 J1 − 1.0 J2", INK), ("     = (0.15, 0.43) m/s", INK)]
    if k >= 1:
        f = u if k == 1 else 1.0
        _arrow(ax, e, 0.6 * f * J[:, 0], BLUE_LIGHT if faded else BLUE, lw=1.8 if faded else 2.5, z=6)
    if k >= 3:
        f = u if k == 3 else 1.0
        _arrow(ax, e, 0.6 * f * J[:, 1], "#f5b79d" if faded else ORANGE, lw=1.8 if faded else 2.5, z=6)
    if k >= 5:
        f = u if k == 5 else 1.0
        c1, c2 = qd[0] * J[:, 0], qd[1] * J[:, 1]
        _arrow(ax, e, f * c1, BLUE, z=8)
        if f > 0.5:
            _arrow(ax, e + c1, (2 * f - 1) * c2, ORANGE, z=8)
        if k > 5:
            _arrow(ax, e, v, INK, lw=3, z=9)
    if k >= 6:
        lines += [("", INK2), ("Answer", INK), ("speed = 0.46 m/s", INK), ("det J = 0.69, not singular", INK)]
    tip(ax, P[2], color=INK, z=10)
    for n_, (txt, col) in enumerate(lines):
        ax.figure.text(0.05, 0.80 - 0.046 * n_, txt, fontsize=9.5, color=col, va="center", family="DejaVu Sans Mono")


example_jacobian.frames = 270


# 13 ---------------------------------------------- solved problem: spatial arm with matrices
RED = "#e34948"


def dh(theta, d, a, alpha):
    ct, st_, ca, sa = np.cos(theta), np.sin(theta), np.cos(alpha), np.sin(alpha)
    return np.array([[ct, -st_ * ca, st_ * sa, a * ct], [st_, ct * ca, -ct * sa, a * st_], [0, sa, ca, d], [0, 0, 0, 1.0]])


def spatial_arm():
    """Numbers for the spatial 3R example: link transforms, cumulative transforms, Jacobian, velocity."""
    th = np.radians([30.0, 60.0, -90.0])
    A = [dh(th[0], 0.4, 0.0, np.pi / 2), dh(th[1], 0.0, 0.5, 0.0), dh(th[2], 0.0, 0.4, 0.0)]
    T = [np.eye(4)]
    for Ai in A:
        T.append(T[-1] @ Ai)
    pe = T[3][:3, 3]
    J = np.column_stack([np.cross(T[j][:3, 2], pe - T[j][:3, 3]) for j in range(3)])
    qd = np.array([0.5, -0.4, 0.8])
    return A, T, J, qd, J @ qd


def _mat(name, M, fmt="{:5.2f}"):
    rows = ["[ " + "  ".join(fmt.format(x + 0.0) for x in r) + " ]" for r in np.round(M, 2)]
    return [name] + rows


def example_spatial(ax, i, st):
    A, T, J, qd, v = spatial_arm()
    k, u = _stage(i, [25, 40, 40, 40, 55, 50, 90])
    fig = ax.figure
    ax.cla()
    ax.set_axis_off()
    ax.set_facecolor(SURFACE)
    ax.set_xlim(-0.15, 0.75)
    ax.set_ylim(-0.3, 0.6)
    ax.set_zlim(0.0, 0.9)
    ax.set_box_aspect((1, 1, 1), zoom=1.25)
    ax.view_init(elev=22, azim=-52 + 12 * np.sin(2 * np.pi * i / example_spatial.frames))
    fig.texts.clear()
    fig.text(0.05, 0.93, "Solved problem: a spatial arm with matrices", fontsize=15, fontweight="bold", color=INK, va="center")
    fig.text(0.05, 0.875, "3R arm from a DH table, θ = (30°, 60°, −90°). Find the tip pose, then its velocity.",
             fontsize=10.5, color=INK2, va="center")
    for g in np.linspace(-0.2, 0.7, 4):  # floor grid
        ax.plot([g + 0.05, g + 0.05], [-0.3, 0.6], [0, 0], color=GRID, lw=1)
        ax.plot([-0.15, 0.75], [g - 0.1, g - 0.1], [0, 0], color=GRID, lw=1)
    P = np.array([Ti[:3, 3] for Ti in T])
    ax.plot(P[:, 0], P[:, 1], 0 * P[:, 2], color=GRID, lw=5, solid_capstyle="round")  # shadow on the floor
    shown = min(max(k, 0), 3)
    ax.plot(P[:, 0], P[:, 1], P[:, 2], color=GRID, lw=5, solid_capstyle="round")
    upto = P[: shown + 1].copy()
    if 1 <= k <= 3:
        upto[-1] = P[k - 1] + u * (P[k] - P[k - 1])
    ax.plot(upto[:, 0], upto[:, 1], upto[:, 2], color=LINK, lw=6, solid_capstyle="round")
    for j in range(shown + 1):  # coordinate frames
        s = 0.16 * (u if (j == k and 1 <= k <= 3) else 1.0)
        o = T[j][:3, 3]
        for c, col in enumerate((RED, AQUA, BLUE)):
            e = o + s * T[j][:3, c]
            ax.plot([o[0], e[0]], [o[1], e[1]], [o[2], e[2]], color=col, lw=2.2)
        ax.text(o[0] - 0.07, o[1] - 0.07, o[2] + 0.04, "{%d}" % j, color=INK2, fontsize=9)
    ax.scatter(*P[3], s=45, color=INK, depthshade=False)
    lines = []
    if k == 0:
        lines = [("DH table (m, rad)", INK), (" i   θ      d    a     α", INK2), (" 1   θ1    0.4   0    90°", INK2),
                 (" 2   θ2     0   0.5    0", INK2), (" 3   θ3     0   0.4    0", INK2), ("", INK2),
                 ("Axes: x red, y green, z blue", INK2)]
    elif k <= 3:
        lines = [(f"1  Link transform {k} of 3", INK), (f"   frame {k - 1} → frame {k}", INK2), ("", INK2)]
        lines += [(s, INK2) for s in _mat(f"A{k} =", A[k - 1])]
    if k >= 4:
        m = _mat("2  T = A1 A2 A3 =", T[3])
        lines = [(m[0], INK)] + [(s, INK2) for s in m[1:]] + [("   last column: tip position", BLUE)]
    if k >= 5:
        f = u if k == 5 else 1.0
        for j, col in enumerate((BLUE, ORANGE, AQUA)):
            d = 0.45 * f * J[:, j]
            ax.quiver(*P[3], *d, color=col if k == 5 else col, lw=2, arrow_length_ratio=0.22, alpha=1.0 if k == 5 else 0.45)
        m = _mat("3  J = [ z × (p − p_i) ] =", J)
        lines += [("", INK2), (m[0], INK)] + [(s, INK2) for s in m[1:]]
    if k >= 6:
        ax.quiver(*P[3], *(0.6 * u * v), color=INK, lw=3, arrow_length_ratio=0.25)
        lines += [("", INK2), ("4  v = J q̇,  q̇ = (0.5, −0.4, 0.8)", INK), (f"   = ({v[0]:.2f}, {v[1]:.2f}, {v[2]:.2f}) m/s", INK),
                  (f"   speed = {np.linalg.norm(v):.2f} m/s", INK)]
    for n, (txt, col) in enumerate(lines):
        fig.text(0.05, 0.80 - 0.046 * n, txt, fontsize=9.5, color=col, va="center", family="DejaVu Sans Mono")


example_spatial.frames = 340
example_spatial.is3d = True


# 14 ---------------------------------------------- teaser banner
def teaser(fig, i, st):
    L = np.array([1.0, 1.0])
    n = teaser.frames
    if "q" not in st:
        s = 2 * np.pi * np.arange(n) / n
        st["p"] = np.c_[1.05 + 0.55 * np.cos(s), 0.55 + 0.45 * np.sin(2 * s)]
        st["q"] = np.array([ik2(1, 1, pt, 1.0) for pt in st["p"]])
    p, q = st["p"], st["q"]
    qdeg = np.degrees(q)
    j = (i + 1) % n
    fig.clf()
    fig.patch.set_facecolor(SURFACE)
    tail = [(i - k) % n for k in range(40, -1, -1)]

    def panel(rect, title, xl, yl, xlab, ylab):
        a = fig.add_axes(rect)
        a.set_facecolor(SURFACE)
        a.set_xlim(*xl)
        a.set_ylim(*yl)
        a.set_xticks([])
        a.set_yticks([])
        for sp in a.spines.values():
            sp.set_color(GRID)
            sp.set_linewidth(1.5)
        a.set_xlabel(xlab, color=INK2, fontsize=11, labelpad=4)
        a.set_ylabel(ylab, color=INK2, fontsize=11, labelpad=4, rotation=0, ha="right", va="center")
        fig.text(rect[0], rect[1] + rect[3] + 0.035, title, fontsize=14, fontweight="bold", color=INK, va="bottom")
        return a

    # joint space
    lo, hi = qdeg.min(0), qdeg.max(0)
    pad = 0.18 * (hi - lo)
    a = panel([0.065, 0.2, 0.2, 0.52], "Joint space", (lo[0] - pad[0], hi[0] + pad[0]), (lo[1] - pad[1], hi[1] + pad[1]), "θ1", "θ2")
    a.plot(np.r_[qdeg[:, 0], qdeg[0, 0]], np.r_[qdeg[:, 1], qdeg[0, 1]], color=GRID, lw=3)
    a.plot(qdeg[tail, 0], qdeg[tail, 1], color=BLUE_LIGHT, lw=3, solid_capstyle="round")
    dq = (qdeg[j] - qdeg[i]) * 12
    a.annotate("", xy=qdeg[i] + dq, xytext=qdeg[i], arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2.2, mutation_scale=14, shrinkA=0, shrinkB=0))
    a.scatter(*qdeg[i], s=90, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=5)
    # arm
    c = fig.add_axes([0.335, 0.08, 0.33, 0.74])
    c.set_facecolor(SURFACE)
    c.set_xlim(-0.75, 2.25)
    c.set_ylim(-0.55, 1.55)
    c.set_aspect("equal")
    c.axis("off")
    ground(c, 0, -0.12)
    c.plot(np.r_[p[:, 0], p[0, 0]], np.r_[p[:, 1], p[0, 1]], color=GRID, lw=3, zorder=1)
    P = fk(L, q[i])
    arm(c, P, lw=8)
    c.scatter(P[:2, 0], P[:2, 1], s=80, color=SURFACE, edgecolor=BLUE, linewidth=2.5, zorder=7)
    tip(c, P[-1], color=ORANGE)
    # task space
    b = panel([0.745, 0.2, 0.2, 0.52], "Task space", (0.25, 1.85), (-0.2, 1.3), "x", "y")
    b.plot(np.r_[p[:, 0], p[0, 0]], np.r_[p[:, 1], p[0, 1]], color=GRID, lw=3)
    b.plot(p[tail, 0], p[tail, 1], color="#f5b79d", lw=3, solid_capstyle="round")
    dp = (p[j] - p[i]) * 12
    b.annotate("", xy=p[i] + dp, xytext=p[i], arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=2.2, mutation_scale=14, shrinkA=0, shrinkB=0))
    b.scatter(*p[i], s=90, color=ORANGE, edgecolor=SURFACE, linewidth=2, zorder=5)
    # mapping arrows across the banner
    ar = dict(arrowstyle="-|>", color=INK2, lw=1.8, mutation_scale=16)
    fig.add_artist(matplotlib.patches.FancyArrowPatch((0.285, 0.60), (0.345, 0.60), transform=fig.transFigure, **ar))
    fig.add_artist(matplotlib.patches.FancyArrowPatch((0.655, 0.60), (0.715, 0.60), transform=fig.transFigure, **ar))
    fig.add_artist(matplotlib.patches.FancyArrowPatch((0.715, 0.34), (0.655, 0.34), transform=fig.transFigure, **ar))
    fig.add_artist(matplotlib.patches.FancyArrowPatch((0.345, 0.34), (0.285, 0.34), transform=fig.transFigure, **ar))
    fig.text(0.5, 0.93, "Forward kinematics   x = f(q)", ha="center", va="center", fontsize=13, color=INK, fontweight="bold")
    fig.text(0.5, 0.055, "Inverse kinematics   q = f⁻¹(x)", ha="center", va="center", fontsize=13, color=INK, fontweight="bold")
    fig.text(0.315, 0.665, "FK", ha="center", fontsize=10.5, color=INK2)
    fig.text(0.685, 0.665, "FK", ha="center", fontsize=10.5, color=INK2)
    fig.text(0.315, 0.255, "IK", ha="center", fontsize=10.5, color=INK2)
    fig.text(0.685, 0.255, "IK", ha="center", fontsize=10.5, color=INK2)
    fig.text(0.165, 0.045, "joint velocity  q̇", ha="center", fontsize=10.5, color=BLUE)
    fig.text(0.845, 0.045, "tip velocity  ẋ = J(q) q̇", ha="center", fontsize=10.5, color=ORANGE)


teaser.frames = 160
teaser.wholefig = True
teaser.figsize = (12.0, 4.4)
teaser.gif_width = 1000
teaser.still = 22


ANIMS = {
    "forward_kinematics": forward_kinematics,
    "inverse_kinematics": inverse_kinematics,
    "numerical_ik": numerical_ik,
    "null_space_motion": null_space_motion,
    "manipulability_ellipse": manipulability,
    "mobile_robot_kinematics": mobile_robot,
    "continuum_robot": continuum,
    "parallel_five_bar": five_bar,
    "jacobian_matrix": jacobian,
    "example_forward_kinematics": example_fk,
    "example_inverse_kinematics": example_ik,
    "example_jacobian": example_jacobian,
    "example_spatial_arm": example_spatial,
    "teaser": teaser,
}


def render(name, fn, preview=False):
    fig = plt.figure(figsize=getattr(fn, "figsize", (7.2, 5.4)), dpi=100, facecolor=SURFACE)
    if getattr(fn, "wholefig", False):
        ax = fig
    elif getattr(fn, "is3d", False):
        ax = fig.add_axes([0.36, 0.0, 0.66, 0.86], projection="3d")
    else:
        ax = fig.add_axes([0.03, 0.02, 0.94, 0.82])
    st = {}
    n = getattr(fn, "frames", N)
    if preview:
        for i in range(n):
            fn(ax, i, st)
            if i in (n // 4, n // 2, n - 1) or (getattr(fn, "is3d", False) and i in (10, 130, 230)):
                fig.savefig(os.path.join(preview, f"{name}_{i:03d}.png"))
        plt.close(fig)
        return
    mp4 = os.path.join(OUT, name + ".mp4")
    w = FFMpegWriter(fps=FPS, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-crf", "20"])
    with w.saving(fig, mp4, dpi=100):
        for i in range(n):
            fn(ax, i, st)
            w.grab_frame(facecolor=SURFACE)
            if i == getattr(fn, "still", -1):
                fig.savefig(os.path.join(OUT, name + ".png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", mp4, "-vf",
                    "fps=20,scale=%d:-1:flags=lanczos," % getattr(fn, "gif_width", 640) + "split[a][b];[a]palettegen=max_colors=64[p];[b][p]paletteuse=dither=none",
                    "-loop", "0", os.path.join(OUT, name + ".gif")], check=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    prev = None
    if args and args[0] == "--preview":
        prev, args = args[1], args[2:]
        os.makedirs(prev, exist_ok=True)
    for name in args or ANIMS:
        render(name, ANIMS[name], prev)
        print("done", name)
