# Level 2: Spatial State / Kinematics - Task Summary

**Total Tasks**: 30

## Kinematic Categories

### 1. Linear Motion - Constant Velocity (5 tasks)
**Physics**: Inertia, Newton's 1st law, frictionless surfaces

- `air_puck_collision` - Frictionless air hockey puck sliding at constant velocity
- `bowling_inertia` - Heavy ball rolling with constant velocity
- `motorized_camera_dolly` - Cart moving at constant speed on track
- `sliding_block_frictionless` - Block sliding on frictionless surface
- `magnetic_sled` - Sled moving via magnetic repulsion

---

### 2. Linear Motion - Acceleration/Deceleration (2 tasks)
**Physics**: Friction, elastic potential energy, F=ma

- `hockey_puck_friction_stop` - Puck decelerating on rough surface due to friction
- `cart_spring_release` - Cart accelerated by compressed spring release

---

### 3. Free Fall (2 tasks)
**Physics**: Gravity, air resistance, terminal velocity

- `elevator_cable_snap` - Free fall under gravity
- `falling_anvil_air` - Free fall with air resistance effects

---

### 4. Inclined Plane Motion (2 tasks)
**Physics**: Gravity components, rolling vs sliding, friction

- `block_on_ramp` - Block sliding down smooth low-friction ramp
- `rolling_sphere_inclined_plane` - Sphere rolling down incline (rotation + translation)

---

### 5. Projectile Motion (3 tasks)
**Physics**: Parabolic trajectories, 2D kinematics, conservation of energy

- `projectile_horizontal_launch` - Object launched horizontally
- `treadmill_projectile` - Projectile motion on moving reference frame
- `ball_rolls_off_cliff` - Ball rolls off table edge (rolling → projectile transition)

---

### 6. Oscillatory Motion - Pendulums (3 tasks)
**Physics**: Simple harmonic motion, periodic motion, restoring forces

- `frictionless_pendulum` - Simple pendulum oscillating without damping
- `conical_pendulum` - Pendulum moving in circular path
- `oscillating_metronome_arm` - Periodic back-and-forth motion

---

### 7. Oscillatory Motion - Springs & Torsion (2 tasks)
**Physics**: Hooke's law, harmonic oscillation, restoring torque

- `spring_oscillation` - Mass oscillating on vertical spring
- `torsional_oscillation` - Object oscillating via rotational restoring force

---

### 8. Circular & Constrained Path Motion (4 tasks)
**Physics**: Centripetal force, banking, spiral dynamics, constrained motion

- `tethered_ball_circular` - Ball moving in circular path via string constraint
- `funnel_marble_vortex` - Marble spiraling in funnel
- `marble_track_loop` - Marble rolling through vertical loop
- `marble_helix_descent` - Marble descending helical spiral track

---

### 9. Rotational Dynamics (4 tasks)
**Physics**: Angular momentum, rotational inertia, gyroscopic effects, friction

- `spinning_gyroscope_precession` - Gyroscope precession under gravity
- `yoyo_drop` - Yo-yo descending via unwinding string
- `spinning_coin_wobble` - Coin spinning with increasing wobble amplitude
- `bicycle_wheel_freespin` - Wheel spinning down due to bearing friction

---

### 10. Elastic Collision (1 task)
**Physics**: Coefficient of restitution, energy dissipation, repeated impacts

- `bouncing_ball_rebound` - Ball bouncing with decreasing height

---

### 11. Coupled Systems (2 tasks)
**Physics**: Momentum transfer, mechanical advantage, fluid dynamics

- `unbalanced_atwood` - Two-mass pulley system with different weights
- `water_deflection` - Fluid momentum and deflection

---

## Summary by Physics Concept

| Physics Concept | Count | Example Tasks |
|----------------|-------|---------------|
| **Inertia & Constant Velocity** | 5 | air_puck, bowling_inertia |
| **Friction & Energy Dissipation** | 3 | hockey_puck_friction_stop, spinning_coin_wobble, bicycle_wheel_freespin |
| **Gravity & Free Fall** | 5 | elevator_cable_snap, falling_anvil_air, projectiles |
| **Harmonic Oscillation** | 5 | pendulums, springs, torsional |
| **Rotational Dynamics** | 6 | gyroscope, yoyo, coin, wheel, rolling objects |
| **Elastic Collisions** | 1 | bouncing_ball_rebound |
| **Constrained Motion** | 7 | tracks, loops, spirals, tethered |
| **Energy Storage/Release** | 2 | spring_oscillation, cart_spring_release |
| **Coupled/Complex Systems** | 2 | atwood, water_deflection |

---

## Task Difficulty Factors

### Easy (Clear Single Trajectory)
- Free fall: `elevator_cable_snap`
- Constant velocity: `air_puck_collision`, `sliding_block_frictionless`
- Simple ramp: `block_on_ramp`

### Medium (Time-Dependent State)
- Oscillations: pendulums, springs (requires predicting phase)
- Deceleration: `hockey_puck_friction_stop`, `bicycle_wheel_freespin`
- Projectiles: parabolic path prediction

### Hard (Complex Paths or Multiple Phases)
- `ball_rolls_off_cliff` - Rolling → projectile transition
- `marble_helix_descent` - 3D helical constraint
- `spinning_coin_wobble` - Instability and tipping dynamics
- `bouncing_ball_rebound` - Multiple collision cycles
- `yoyo_drop` - Coupled rotation and translation

---

## Common Kinematic Variables Tested

- **Position** (x, y, z): All tasks
- **Velocity**: Constant velocity tasks, projectiles, oscillations
- **Angular position**: Pendulums, rotational tasks
- **Angular velocity**: Spinning objects, rolling objects
- **Phase of oscillation**: Spring, pendulum, metronome
- **Height/Vertical position**: Free fall, inclines, bouncing
- **Distance traveled**: Sliding, rolling, deceleration tasks

---

## Key Physical Principles

1. **Newton's Laws**: F=ma, inertia, action-reaction
2. **Conservation of Energy**: Potential ↔ kinetic conversions
3. **Conservation of Momentum**: Collisions, coupled systems
4. **Friction**: Kinetic friction, rolling resistance, air drag
5. **Gravity**: Constant acceleration, projectile motion
6. **Oscillation**: Restoring forces, periodic motion
7. **Rotation**: Angular momentum, moment of inertia, gyroscopic effects
8. **Elastic Properties**: Spring forces, bouncing, energy loss

---

**Last Updated**: 2026-02-19
