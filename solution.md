# Q2. Analysis of Observation State, Action State, and Reward Function

## (1) Observation State Contents and Location

### Observation State Contents
In the training example, the observation state uses kinematic information (`ObservationType.KIN`) and contains:

#### Base State Vector (12 dimensions):
```python
obs_12[i, :] = np.hstack([obs[0:3], obs[7:10], obs[10:13], obs[13:16]])
```

The 12-dimensional base observation includes:
- **Position** (3D): `obs[0:3]` → (x, y, z) coordinates in world frame
- **Euler Angles** (3D): `obs[7:10]` → (roll, pitch, yaw) orientation in radians
- **Linear Velocity** (3D): `obs[10:13]` → (vx, vy, vz) velocity in world frame
- **Angular Velocity** (3D): `obs[13:16]` → (wx, wy, wz) angular velocity in body frame

#### Action History Buffer:
```python
for i in range(self.ACTION_BUFFER_SIZE):  # ACTION_BUFFER_SIZE = ctrl_freq//2
    ret = np.hstack([ret, np.array([self.action_buffer[i][j, :] for j in range(self.NUM_DRONES)])])
```

- **Buffer size**: `ctrl_freq//2` = 15 time steps (0.5 seconds of history)
- **Action dimensions**: 4 dimensions per time step (for RPM mode)
- **Total action history**: $15 \times 4 = 60$ dimensions

#### Total Observation Dimensions:
**72 dimensions** = $12$ (base state) + $60$ (action history)

## (2) Action State Explanation

### File Location and Definition
The action state is defined in:
- **File**: `gym_pybullet_drones/envs/BaseRLAviary.py`
  - Method `_actionSpace()` (lines ~110-130)
  - Method `_preprocessAction()` (lines ~130-220)
- **Configuration**: `gym_pybullet_drones/examples/learn.py`
  - `DEFAULT_ACT = ActionType('rpm')` (line 36)

### Action State Meaning
In this context, the action state refers to the **RPM control mode** with the following characteristics:

#### Action Space Definition:
```python
if self.ACT_TYPE in [ActionType.RPM, ActionType.VEL]:
    size = 4  # Four propellers
act_lower_bound = np.array([-1*np.ones(size) for i in range(self.NUM_DRONES)])
act_upper_bound = np.array([+1*np.ones(size) for i in range(self.NUM_DRONES)])
```

- **Dimensions**: 4D action vector $\mathbf{a} = [a_1, a_2, a_3, a_4]^T$
- **Range**: Each $a_i \in [-1, +1]$
- **Physical meaning**: RPM adjustment coefficients for four propellers

#### Action Processing:
```python
if self.ACT_TYPE == ActionType.RPM:
    rpm[k,:] = np.array(self.HOVER_RPM * (1+0.05*target))
```

**Mathematical conversion**:
$$\text{RPM}_i = \text{HOVER\_RPM} \times (1 + 0.05 \times a_i)$$

- When $a_i = -1$: $\text{RPM}_i = 0.95 \times \text{HOVER\_RPM}$ (95% of hover RPM)
- When $a_i = 0$: $\text{RPM}_i = \text{HOVER\_RPM}$ (baseline hover RPM)  
- When $a_i = +1$: $\text{RPM}_i = 1.05 \times \text{HOVER\_RPM}$ (105% of hover RPM)

This design allows fine-grained control with $\pm 5\%$ adjustment around the stable hovering RPM.

## (3) Reward Function Explanation

### Reward Function Implementation
```python
def _computeReward(self):
    state = self._getDroneStateVector(0)
    ret = max(0, 2 - np.linalg.norm(self.TARGET_POS-state[0:3])**4)
    return ret
```

### Reward Function Meaning

#### Target Definition:
```python
self.TARGET_POS = np.array([0,0,1])  # Target position at $(0,0,1)$
```

#### Mathematical Formula:
$$r_t = \max(0, 2 - \|\mathbf{p}_{target} - \mathbf{p}_{current}\|^4)$$

Where:
- $\mathbf{p}_{target} = [0, 0, 1]^T$ (hovering target)
- $\mathbf{p}_{current} = \text{state}[0:3]$ (drone's current position)
- $\|\cdot\|$ denotes Euclidean distance

#### Reward Characteristics:

1. **Maximum reward**: $2.0$ (when drone is exactly at target position)
2. **Distance penalty**: Uses fourth power ($\cdot^4$) for aggressive penalization
3. **Zero reward threshold**: When distance $> 2^{1/4} \approx 1.19$ meters
4. **Reward range**: $[0, 2]$

#### Reward Behavior Analysis:
- **At target** (distance $= 0$): $r_t = 2.0$
- **Close to target** (distance $= 0.5$): $r_t = 2 - (0.5)^4 = 2 - 0.0625 = 1.9375$
- **Moderate distance** (distance $= 1.0$): $r_t = 2 - (1.0)^4 = 2 - 1 = 1.0$  
- **Far from target** (distance $= 1.2$): $r_t = \max(0, 2 - (1.2)^4) = \max(0, 2 - 2.07) = 0$

#### Design Purpose:
The fourth-power penalty creates a **steep reward gradient** that:
- Strongly encourages precise positioning near the target
- Rapidly diminishes reward as the drone moves away from target
- Promotes stable hovering behavior rather than wandering
- Provides clear learning signal for position control

This reward function effectively guides the reinforcement learning agent to learn precise hovering control at the specified target position.