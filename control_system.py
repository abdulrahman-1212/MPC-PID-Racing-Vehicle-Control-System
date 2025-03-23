""" 
ToDos
    0. solve control problem
    1. Work Documentation (Technical Report & README file on the repo).
    2. Hyperparameter Tuning.
    3. Complete Odom_callback and path_callback.
    4. Test on a Trajectory (different conditions).
    5. Convert Code to C++.
    6. Take acceleration gradient (jerk) into consideration in calcauting throttle gain.
    7. Try Coupled Control (Try to give a higher priority for Steering over Throttling by taking steering into conisderation in the calculations of throttle gain.
    This is supposed to enhance the performance at curves, but will affect vehicle speed.)

Hyperparameters:
    - PID (Kp, Ki, Kd)
    - MPC Prediciton Horizon
    - numsteps in control_loop and dt (frequency of updating state)

Vehicle Constants:
    - Wheelbase (1m)
    - max acceleration and decceleration ([-3: 3] m/s^2, -ve for backward motion, 0 to stop)
    - max velocity
    - max steering ([-pi/4: pi/4])
    
ROS Channels:
    - Odometry Subscriber: to get location feedback from odometry sensors
        - Topic name: /carla/ego_vehicle/odometry (depends on the simulator)
        - Message Type: Odometry
        - Callback Function: odom_callback
        
    - Path Subscriber: to get the next desired state (x, y, v, yaw)
        - Topic name: /path (need to communicate with path planning team)
        - Message Type: Path (Custom Message built by Path Planning team not a ROS-built-in message)
        - Callback Function: path_callback
        
    - Throttle Pub: publish throttle to vehicle model
        - Topic name: /throttle
        - Message Type: Float64
        
    - Steering Pub: publish steering to vehicle model 
        - Topic name: /steering
        - Message Type: Float64 

The Control Process Steps (Control Loop):
    1. Get the next desired state (x, y, v, yaw)
    2. Pass the desired x, y, and v to the PID Controller to compute throttle command.
    3. Pass the the desired state to the MPC to compute steering command.
    4. Pass throttle and steering commands to the Bicycle Model to update the state.
"""

import numpy as np
import matplotlib.pyplot as plt
import casadi as ca
import matplotlib.animation as animation
import time

import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64
# Control System
class ControlSystem:
    def __init__(self, desired_path: list=[], setpoint_velocity: float=10, setpoint_yaw: float=0.5):
        self.bicycle_model = KinematicBicycleModel(dt=0.1)

        # self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        # self.path_sub = rospy.Subscriber('/goal_pose', PoseStamped, self.path_callback)
        # self.throttle_pub = rospy.Publisher('/throttle', Float64, queue_size=10)
        # self.steering_pub = rospy.Publisher('/steering', Float64, queue_size=10)

        self.path = desired_path                # [(x1, y1, v1, yaw1), (x2, y2, v2, yaw2), ...]
        # self.target_pose = []
        self.pid = CascadedPID(kp_pos=0.9, ki_pos=0.1, kd_pos=0.01, kp_vel=1, ki_vel=0.5, kd_vel=0.02)  # PID constants for throttle control
        self.mpc = MPC(horizon=10, dt=0.1, wheelbase=1)  # MPC for steering control
        
        
    def path_callback(self, msg):
        quaternion = (
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w
        )
        yaw = 2 * np.arctan2(quaternion[2], quaternion[3])
        self.target_pose = [
            msg.pose.position.x,
            msg.pose.position.y,
            10.0,  # Default desired velocity (could be parameterized)
            yaw
        ]
        rospy.loginfo(f"Received new target: {self.target_pose}")


    def odom_callback(self, msg):
        quaternion = (
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        )
        yaw = 2 * np.arctan2(quaternion[2], quaternion[3])
        self.current_pose = [
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.twist.twist.linear.x,
            yaw
        ]
        self.bicycle_model.set_state(self.current_pose)
    
    
    def control_loop(self):
        tolerance = 0.1
        num_steps = 30
        dt = 0.1
        max_iters = 10
        iters_count = 0
        for i in range(len(self.path)):
            while True:
                iters_count += 1
                if iters_count > max_iters:
                    print("Reached Max Iterations!")
                    break 
                # Get the desired state (x, y, v, yaw)
                desired_pose_x, desired_pose_y, desired_vel, desired_yaw = self.desired_path[i]

                # Get the current state from the model
                current_vel = self.bicycle_model.get_state()[2]
                current_pose = self.bicycle_model.get_state()[0:2]

                # Compute throttle and steering
                throttle_output = self.pid.compute(
                    [desired_pose_x, desired_pose_y],
                    current_pose,
                    desired_vel,
                    current_vel
                )
                steering_output = self.mpc.compute(self.bicycle_model.state, desired_yaw)
                # publish steering and throttle
                # self.steering_pub.publish(steering_output)
                # self.throttle_pub.publsih(throttle_output)
                # Update the bicycle model
                self.bicycle_model.set_throttle(throttle_output)
                self.bicycle_model.set_steering(steering_output)
                self.bicycle_model.compute_state()

                # Check if the target is reached
                x_error = np.abs(desired_pose_x - current_pose[0])
                y_error = np.abs(desired_pose_y - current_pose[1])

                # Termination Conditions
                if (x_error < tolerance and y_error < tolerance):
                    print("Target reached!")
                    break

        # Sleep for dt seconds (to match real-time)
        rospy.time.sleep(self.bicycle_model.dt)

        return self.bicycle_model.get_history()

    def plot_results(self):
        """Plot the results (trajectory, velocity, yaw) from the bicycle model."""
        self.bicycle_model.plot_trajectory()
        self.bicycle_model.plot_velocity()
        self.bicycle_model.plot_yaw()
        plt.show()


# PID Controller
class PID:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.last_error = 0
        self.error_sum = 0
        self.last_time = time.time()

    def compute(self, desired_vel, current_vel, desired_pose, current_pose):
        pos_tolerance = 1e-3  # Small tolerance for floating-point comparison
        vel_tolerance = 0.1
        current_time = time.time()
        dt = current_time - self.last_time
        if dt == 0:
            dt = 0.1  # Avoid division by zero
            
        if np.linalg.norm(np.array(desired_pose) - np.array(current_pose)) < pos_tolerance:
            output = 0
            self.last_time = current_time
            return output

        elif (np.linalg.norm(np.array(desired_vel) - np.array(current_vel)) < vel_tolerance):
            error = self.last_error
            self.error_sum += error * dt
            delta_error = error - self.last_error
            output = self.kp * error + self.ki * self.error_sum + self.kd * (delta_error / dt)

            self.last_error = error
            self.last_time = current_time
            return output
        

        vel_error = desired_vel - current_vel
        pose_error = np.sqrt((desired_pose[0] - current_pose[0])**2 + (desired_pose[1] - current_pose[1])**2)
        error = vel_error + pose_error
    
        self.error_sum += error * dt
        delta_error = error - self.last_error
        output = self.kp * error + self.ki * self.error_sum + self.kd * (delta_error / dt)

        self.last_error = error
        self.last_time = current_time

        return output

class CascadedPID:
    def __init__(self,
            kp_pos, ki_pos, kd_pos,
            kp_vel, ki_vel, kd_vel,
        ):
        self.kp_pos = kp_pos
        self.ki_pos = ki_pos
        self.kd_pos = kd_pos

        self.kp_vel = kp_vel
        self.ki_vel = ki_vel
        self.kd_vel = kd_vel

        self.last_error_pos = 0
        self.last_error_vel = 0
        self.sum_error_pos = 0
        self.sum_error_vel = 0

        self.last_time = time.time()
    
    def compute(self, desired_pos, current_pos, desired_vel, current_vel):
        current_time = time.time()
        dt = current_time - self.last_time
        if dt == 0:
            dt = 0.1        # Avoid division by zero

        error_pos = np.linalg.norm(np.array(current_pos) - np.array(desired_pos))
        self.sum_error_pos += error_pos * dt
        delta_error_pos = error_pos - self.last_error_pos
        desired_vel_adjusted = (
            self.kp_pos * error_pos +
            self.ki_pos * self.sum_error_pos +
            self.kd_pos * (delta_error_pos / dt)
        )
        self.last_error_pos = error_pos

        error_vel = desired_vel_adjusted - current_vel
        self.sum_error_vel += error_vel * dt
        delta_error_vel = error_vel - self.last_error_vel
        throttle = (
            self.kp_vel * error_vel + 
            self.ki_vel * self.sum_error_vel +
            self.kd_vel * delta_error_vel / dt
        )
        self.last_error_vel = error_vel
        self.last_time = current_time
        return throttle

class MPC:
    def __init__(self, horizon, dt, wheelbase):
        self.horizon = horizon
        self.dt = dt
        self.wheelbase = wheelbase
        self.opti = ca.Opti()  # Optimization problem

        # Define optimization variables
        self.steering = self.opti.variable(self.horizon)
        self.states = self.opti.variable(4, self.horizon)  # [x, y, v, yaw]

        # Define parameters
        self.current_state = self.opti.parameter(4)  # Current state of the vehicle
        self.setpoint_yaw = self.opti.parameter()  # Desired yaw angle

        # Objective function
        self.cost = 0
        for t in range(self.horizon):
            self.cost += (self.states[3, t] - self.setpoint_yaw) ** 2 # Penalize yaw error
            self.cost += self.steering[t] ** 2  # Penalize large steering inputs
        self.opti.minimize(self.cost)

        # Dynamics constraints
        for t in range(self.horizon - 1):
            x_next = self.states[0, t] + self.states[2, t] * ca.cos(self.states[3, t]) * self.dt
            y_next = self.states[1, t] + self.states[2, t] * ca.sin(self.states[3, t]) * self.dt
            v_next = self.states[2, t]  # Constant velocity for simplicity
            yaw_next = self.states[3, t] + (self.states[2, t] * ca.tan(self.steering[t]) / self.wheelbase) * self.dt

            self.opti.subject_to(self.states[0, t + 1] == x_next)
            self.opti.subject_to(self.states[1, t + 1] == y_next)
            self.opti.subject_to(self.states[2, t + 1] == v_next)
            self.opti.subject_to(self.states[3, t + 1] == yaw_next)

        # Steering angle constraints
        self.opti.subject_to(self.opti.bounded(-np.pi / 4, self.steering, np.pi / 4))

        # Initial state constraint
        self.opti.subject_to(self.states[:, 0] == self.current_state)

        # Solver settings
        self.opti.solver('ipopt')

    def compute(self, current_state, setpoint_yaw):
        self.opti.set_value(self.current_state, current_state)
        self.opti.set_value(self.setpoint_yaw, setpoint_yaw)

        sol = self.opti.solve()
        return sol.value(self.steering[0])  # Return the first steering angle



# Kinematic Bicycle Model
class KinematicBicycleModel:
    def __init__(self, dt=0.1, throttle = 0, steering = 0):
        self.dt = dt                    # Time step (seconds)
        self.state = [0, 0, 0, 0]  # Initial state [x, y, velocity, yaw]
        self.L = 1                      # Wheelbase (meters)
        self.a_max = 3                  # Max acceleration (m/s^2)
        self.v_max = 10  # Max velocity (m/s)
        self.yaw_rate_max = np.pi
        self.history = []               # To store the history of states
        self.throttle = throttle
        self.steering = steering
        
        self.mass = 1000                # Mass of the vehicle (kg)
        self.C_rr = 0.01                # Rolling resistance coefficient
        self.C_d = 0.3                  # Air drag coefficient
        self.A = 2.0                    # Frontal area (m^2)
        self.rho = 1.225                # Air density (kg/m^3)

        
    def set_throttle(self, t):
        self.throttle = t
    def set_steering(self, s):
        self.steering = s
                
    def log_state(self):
        """Log the current state for future analysis."""
        self.history.append(list(self.state))

    def compute_state(self):
        """Computes the next state of the bicycle."""
        x, y, v, yaw = self.state

        # Calculate friction forces
        F_rr = self.C_rr * self.mass * 9.81  # Rolling resistance
        F_drag = 0.5 * self.rho * self.C_d * self.A * v**2  # Air drag
        F_friction = F_rr + F_drag

        # Update velocity based on throttle (acceleration)
        a = self.throttle * self.a_max  # Calculate acceleration
        v += (a - F_friction / self.mass) * self.dt  # Update velocity
        v = np.clip(v, -self.v_max, self.v_max)

        # Update yaw (orientation) based on velocity and steering angle
        yaw_rate = (v * np.tan(self.steering) / self.L)
        yaw_rate = np.clip(yaw_rate, -self.yaw_rate_max, self.yaw_rate_max)
        yaw += yaw_rate * self.dt

        # Update position based on current velocity and yaw
        x += v * np.cos(yaw) * self.dt
        y += v * np.sin(yaw) * self.dt
        
        # Update the state
        self.state = [x, y, v, yaw]
        self.log_state()  # Log the new state
        return self.state
    
    def get_state(self):
        return self.state
    
    def set_state(self, new_state):
        self.state = new_state
    
    def get_history(self):
        return self.history

    def plot_trajectory(self):
        """Plots the x-y trajectory of the bicycle."""
        history = np.array(self.history)
        plt.figure(figsize=(10, 6))
        plt.plot(history[:, 0], history[:, 1], label="Trajectory (x, y)", color='b')
        plt.xlabel("X Position (m)")
        plt.ylabel("Y Position (m)")
        plt.title("Bicycle Trajectory")
        plt.grid(True)
        plt.legend()

    def plot_velocity(self):
        """Plots the velocity over time."""
        velocity = np.array(self.history)[:, 2]  # Extract velocity from the history
        time = np.arange(len(velocity)) * self.dt
        plt.figure(figsize=(10, 6))
        plt.plot(time, velocity, label="Velocity (m/s)", color='g')
        plt.xlabel("Time (s)")
        plt.ylabel("Velocity (m/s)")
        plt.title("Velocity Over Time")
        plt.grid(True)
        plt.legend()

    def plot_yaw(self):
        """Plots the yaw over time."""
        yaw = np.array(self.history)[:, 3]
        time = np.arange(len(yaw)) * self.dt
        plt.figure(figsize=(10, 6))
        plt.plot(time, yaw, label="Yaw (rad)", color='g')
        plt.xlabel("Time (s)")
        plt.ylabel("Yaw (rad)")
        plt.title("Yaw Over Time")
        plt.grid(True)
        plt.legend()



if __name__ == '__main__':    
    # Initialize the ControlSystem with the circular trajectory
    desired_path = [(np.cos(t), np.sin(t), 10, np.arctan2(np.sin(t), np.cos(t))) for t in np.linspace(0, 2*np.pi, 100)]
    # Set up the desired trajectory (example path)
    # desired_path = [
    #     (10, 10, 5, np.pi/4),    # (x, y, velocity, yaw)    
    #     (20, 20, 8, np.pi/4),
    #     (30, 20, 10, 0),
    #     (40, 20, 5, 0),
    #     (50, 20, 0, 0),
    # ]

    # Create the control system and run the control loop
    control_system = ControlSystem(desired_path=desired_path)
    trajectory_history = control_system.control_loop()
    # for point in desired_path:
    #     print(point)
    control_system.plot_results()

