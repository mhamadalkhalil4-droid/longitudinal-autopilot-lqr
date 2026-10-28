# Longitudinal autopilot: pitch-attitude hold (LQR vs PID) and altitude hold

Cascade autopilot for a jet transport in cruise, on the linear longitudinal model
`x = [u, w, q, θ]`, elevator input, altitude integrated as `ḣ = U0·θ − w`.

```
altitude loop (P + I + vertical-speed damping, θ_cmd ≤ ±5°)
   └─ pitch loop: LQR on [u, w, q, θ] + ∫θ-error    — compared with a classical PID + pitch-rate damper
        └─ elevator (±20°)
```

## Open-loop modes

| Mode | ωn [rad/s] | ζ |
|---|---|---|
| Short period | 0.97 | 0.39 |
| Phugoid | 0.067 | 0.05 |

![open loop](figures/open_loop.png)

## Pitch loop — LQR with Bryson weights

Weights from Bryson's rule (q ≤ 0.1 rad/s, θ ≤ 0.05 rad, δe ≤ 0.1 rad). Gains
`K = [0, 0.002, −1.62, −2.22, −0.20]` on `[u, w, q, θ, ∫θ]`; closed-loop short period moves to
ωn ≈ 1.6 rad/s, ζ ≈ 0.7.

![pitch step](figures/pitch_step.png)

## Altitude step of +100 m

| Inner loop | Overshoot | Settling (±2 m) | max θ | max δe |
|---|---|---|---|---|
| LQR + integral | 0.9 m | 23 s | 5.4° | 11° |
| PID + q-damper (hand-tuned) | 0.9 m | 24 s | 4.6° | 7.5° |

Both reach the target without overshoot; the LQR is slightly more aggressive on the elevator because
it also sees `w`. Speed bleeds off by ~6 m/s during the climb because there is no auto-throttle in
the model — that is the phugoid energy exchange, left visible on purpose.

![altitude step](figures/altitude_step.png)

## Run

```bash
pip install numpy scipy matplotlib
python autopilot.py
```

## What I learnt doing this

* My first attempt was a single LQR on the full `[u, w, q, θ, h, ∫h]` state. It produced
  absurd gains (hundreds on θ) and a diving response: because `ḣ = 235·θ`, any altitude weight
  turns into an enormous pitch weight, and with the elevator saturating the loop winds up. The
  cascade with a limited θ command is the textbook fix, and now I understand *why* it is the textbook.
* The pitch-rate damper gain must be **positive** with this sign convention (positive δe gives
  negative q): I had it inverted at first and got 35° pitch excursions. Sign conventions are not
  a detail.

## Next

Lateral-directional (Dutch roll, yaw damper), auto-throttle, gain scheduling over two flight
conditions, and the same design in Simulink for comparison.
