# Action Space Scaling for Robot Control

This document explains the concept and implementation of scaling the normalized action space from the RL policy to the physical action space of the robot.

## The Concept

In this project, the Reinforcement Learning (RL) algorithms (SAC, TD3, DDPG) are designed to output actions in a **normalized range of `[-1, 1]`**. This is a standard practice that makes the algorithms stable and general-purpose.

However, the robot's physical joints can move in a much larger range, for example, from **-pi to +pi radians** (-180 to +180 degrees).

To bridge this gap, we must **scale** the normalized action from the policy to the full physical range of the robot before sending it as a command to the simulator.

-   **Without scaling (the old way):** The robot could only move its joints between -1 and +1 radians (about -57 to +57 degrees). This limited its workspace and made the policy less general.
-   **With scaling (the new, correct way):** The robot can now use its full range of motion, allowing it to solve more complex tasks and learn a more robust policy.

## The Scaling Formula

The standard formula to map a value from a normalized range `[-1, 1]` to a physical range `[min_angle, max_angle]` is:

`scaled_action = min_angle + (normalized_action + 1.0) * 0.5 * (max_angle - min_angle)`

### How it was implemented in `pick_place_env.py`

In our case, `min_angle` is `-pi` and `max_angle` is `+pi`.

```python
# The normalized action from the policy (in range [-1, 1])
action = ... 

# The scaling formula
pi = np.pi
scaled_action = -pi + (action + 1.0) * 0.5 * (pi - -pi)

# 'scaled_action' is now in the range [-pi, +pi] and can be sent to the robot.
```

### Step-by-Step Breakdown of the Formula

1.  **`(action + 1.0)`**: Shifts the `[-1, 1]` range to `[0, 2]`.
2.  **`* 0.5`**: Scales the `[0, 2]` range down to `[0, 1]`, effectively creating a percentage.
3.  **`* (pi - -pi)`**: Scales this `[0, 1]` percentage to the full width of the target physical range, which is `2 * pi`. The range is now `[0, 2 * pi]`.
4.  **`-pi + ...`**: Shifts the `[0, 2 * pi]` range to start at the correct minimum value, resulting in the final `[-pi, +pi]` range.
