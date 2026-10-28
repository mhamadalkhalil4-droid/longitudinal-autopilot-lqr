"""Pitch-attitude and altitude-hold autopilot — longitudinal dynamics of a jet transport in cruise.

Linear model  x = [u, w, q, θ]  (speed, heave, pitch rate, pitch perturbations), input δe [rad].
Altitude is integrated outside the model:  ḣ = U0·θ − w.
Numerical values are representative of a large jet transport at high-subsonic cruise; the point of the
exercise is the control design, not the exact aircraft.

Architecture (cascade):
    altitude loop  (P + I + vertical-speed damping, θ_cmd limited to ±5°)
        -> pitch loop  (LQR on [u, w, q, θ] + integral of θ error)     [compared with a classical PID]
            -> elevator (±20°)
"""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_are

U0, g = 235.0, 9.81
A = np.array([[-0.0069,  0.0139,   0.0,   -g],
              [-0.0905, -0.3149, 235.9,  0.0],
              [ 0.0004, -0.0034, -0.4282, 0.0],
              [ 0.0,     0.0,     1.0,    0.0]])
B = np.array([[-0.0001], [-5.5], [-1.158], [0.0]])
DE_MAX, TH_MAX = np.deg2rad(20), np.deg2rad(5)

# ---------- open-loop modes ----------
print("Open-loop longitudinal modes:")
for e in np.linalg.eigvals(A):
    if e.imag >= 0:
        wn = abs(e); print(f"  ωn = {wn:.3f} rad/s  ζ = {-e.real/wn:.3f}   ({'short period' if wn > 0.5 else 'phugoid'})")

# ---------- inner loop 1: LQR with integral of pitch error ----------
Aa = np.zeros((5, 5)); Aa[:4, :4] = A; Aa[4, 3] = 1.0          # x5 = ∫(θ − θ_cmd)
Ba = np.vstack([B, [[0.0]]])
Q = np.diag([0, 0, 1 / 0.1**2, 1 / 0.05**2, 1 / 0.5**2])      # Bryson: q ≤ 0.1 rad/s, θ ≤ 0.05 rad
R = np.array([[1 / 0.1**2]])                                  #          δe ≤ 0.1 rad
P = solve_continuous_are(Aa, Ba, Q, R)
K = np.linalg.solve(R, Ba.T @ P)[0]
print("LQR gains [u, w, q, θ, ∫θ]:", np.round(K, 3))
print("Closed-loop poles:", np.round(np.linalg.eigvals(Aa - Ba @ K[None, :]), 3))

def lqr_pitch(x, i_th, th_cmd):
    return -(K[:4] @ (x - np.array([0, 0, 0, th_cmd])) + K[4] * i_th)

# ---------- inner loop 2: classical PID pitch + rate damper ----------
Kp, Ki_, Kd, Kq = -1.5, -0.15, 0.0, 1.0      # hand-tuned: θ gain, ∫θ gain, (unused D), pitch-rate damper
def pid_pitch(x, i_th, th_cmd, e_prev, dt):
    e = th_cmd - x[3]
    return Kp * e + Ki_ * (-i_th) + Kd * (e - e_prev) / dt + Kq * x[2]

# ---------- outer loop ----------
Kh, Khd, Kih = 0.002, 0.01, 1e-5
def altitude_cmd(h, hdot, ie):
    return np.clip(Kh * (h_cmd - h) + Kih * ie - Khd * hdot, -TH_MAX, TH_MAX)

def simulate(inner, T=200, dt=0.005, h_step=True, th_step=None):
    n = int(T / dt); x = np.zeros(4); h = i_th = ie = e_prev = 0.0
    out = np.zeros((n, 7))
    for k in range(n):
        t = k * dt; hdot = U0 * x[3] - x[1]
        th_cmd = altitude_cmd(h, hdot, ie) if th_step is None else (th_step if t > 1 else 0.0)
        if inner == "lqr":
            de = lqr_pitch(x, i_th, th_cmd)
        else:
            de = pid_pitch(x, i_th, th_cmd, e_prev, dt); e_prev = th_cmd - x[3]
        de = np.clip(de, -DE_MAX, DE_MAX)
        x = x + (A @ x + B[:, 0] * de) * dt
        i_th += (x[3] - th_cmd) * dt; h += hdot * dt; ie += (h_cmd - h) * dt
        out[k] = t, x[0], x[1], x[2], x[3], h, de
    return out

h_cmd = 100.0
th_lqr = simulate("lqr", T=40, th_step=np.deg2rad(2))
th_pid = simulate("pid", T=40, th_step=np.deg2rad(2))
alt_lqr = simulate("lqr"); alt_pid = simulate("pid")

def report(name, o):
    h = o[:, 5]; ov = h.max() - h_cmd
    inside = np.abs(h - h_cmd) < 2
    ts = o[len(h) - 1 - np.argmax(~inside[::-1]), 0] if inside[-1] else np.nan
    print(f"{name:12s} altitude +100 m: overshoot {ov:5.1f} m   settling (±2 m) {ts:6.1f} s   max |θ| {np.rad2deg(np.abs(o[:,4]).max()):.1f}°   max |δe| {np.rad2deg(np.abs(o[:,6]).max()):.1f}°   Δu at end {o[-1,1]:.1f} m/s")
report("LQR inner", alt_lqr); report("PID inner", alt_pid)

# ---------- plots ----------
t = np.linspace(0, 300, 30000); x = np.zeros(4); Y = np.zeros((len(t), 4))
for k in range(len(t)):
    de = np.deg2rad(-1.0) if t[k] < 2 else 0.0
    x = x + (A @ x + B[:, 0] * de) * (t[1] - t[0]); Y[k] = x
fig, ax = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True)
ax[0].plot(t, np.rad2deg(Y[:, 3])); ax[0].set_ylabel("θ [°]"); ax[0].set_title("Open loop, 1° elevator pulse: short period (first seconds) then phugoid (≈ 90 s period)")
ax[1].plot(t, Y[:, 0]); ax[1].set_ylabel("Δu [m/s]"); ax[1].set_xlabel("t [s]")
for a in ax: a.grid(alpha=.3)
plt.tight_layout(); plt.savefig("figures/open_loop.png", dpi=130)

fig, ax = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True)
ax[0].plot(th_lqr[:, 0], np.rad2deg(th_lqr[:, 4]), label="LQR + integral"); ax[0].plot(th_pid[:, 0], np.rad2deg(th_pid[:, 4]), label="PID + q damper")
ax[0].axhline(2, c="k", ls="--"); ax[0].set_ylabel("θ [°]"); ax[0].legend(); ax[0].set_title("Pitch-attitude step of +2°")
ax[1].plot(th_lqr[:, 0], np.rad2deg(th_lqr[:, 6])); ax[1].plot(th_pid[:, 0], np.rad2deg(th_pid[:, 6])); ax[1].set_ylabel("δe [°]"); ax[1].set_xlabel("t [s]")
for a in ax: a.grid(alpha=.3)
plt.tight_layout(); plt.savefig("figures/pitch_step.png", dpi=130)

fig, ax = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
ax[0].plot(alt_lqr[:, 0], alt_lqr[:, 5], label="LQR inner loop"); ax[0].plot(alt_pid[:, 0], alt_pid[:, 5], label="PID inner loop")
ax[0].axhline(h_cmd, c="k", ls="--"); ax[0].set_ylabel("Δh [m]"); ax[0].legend(); ax[0].set_title("Altitude step of +100 m (θ_cmd limited to ±5°, no auto-throttle)")
ax[1].plot(alt_lqr[:, 0], np.rad2deg(alt_lqr[:, 4])); ax[1].plot(alt_pid[:, 0], np.rad2deg(alt_pid[:, 4])); ax[1].set_ylabel("θ [°]")
ax[2].plot(alt_lqr[:, 0], alt_lqr[:, 1]); ax[2].plot(alt_pid[:, 0], alt_pid[:, 1]); ax[2].set_ylabel("Δu [m/s]"); ax[2].set_xlabel("t [s]")
for a in ax: a.grid(alpha=.3)
plt.tight_layout(); plt.savefig("figures/altitude_step.png", dpi=130)
