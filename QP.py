import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from qpsolvers import solve_qp

class ARM:
    def __init__(self, params):
        self.params = params
        self.q = np.array([np.pi / 2, np.pi / 4])  # initial joint angles
        self.obstacle_position = np.array([0, 1])  # obstacle position
        self.obstacle_radius = 1.5  # obstacle radius

    def jacobian_2dof(self, q):
        """Calculate the Jacobian for a 2-DOF robot arm."""
        l1, l2 = self.params['l1'], self.params['l2']
        q1, q2 = q
        J = np.array([
            [-l1 * np.sin(q1) - l2 * np.sin(q1 + q2), -l2 * np.sin(q1 + q2)],
            [l1 * np.cos(q1) + l2 * np.cos(q1 + q2), l2 * np.cos(q1 + q2)]
        ])
        return J

    def forward_kinematics(self, q):
        """Calculate the position of the end-effector using forward kinematics."""
        l1, l2 = self.params['l1'], self.params['l2']
        q1, q2 = q
        x = l1 * np.cos(q1) + l2 * np.cos(q1 + q2)
        y = l1 * np.sin(q1) + l2 * np.sin(q1 + q2)
        return np.array([x, y])

    def avoid_obstacle(self, q, v_d):
        """Calculate obstacle avoidance constraints."""
        pos_current = self.forward_kinematics(q)
        d_ro = np.linalg.norm(pos_current - self.obstacle_position)
        d_i = self.obstacle_radius + 0.2  # influence distance
        d_s = self.obstacle_radius  # stopping distance

        if d_ro < d_i:
            n_ro = (pos_current - self.obstacle_position) / d_ro  # unit vector
            tangential_direction = np.array([-n_ro[1], n_ro[0]])  # Perpendicular direction
            v_d = v_d + 0.1 * tangential_direction  # Adjust velocity to move around obstacle

            d_dot_ro = np.dot(n_ro, v_d)
            A_obs = np.dot(n_ro, self.jacobian_2dof(q))
            b_obs = (d_ro - d_s) / (d_i - d_s) * d_dot_ro

            return A_obs, b_obs, v_d
        return None, None, v_d

    def update(self, frame):
        J = self.jacobian_2dof(self.q)
        target_position = self.params['target_position']
        pos_current = self.forward_kinematics(self.q)
        v_d = target_position - pos_current
        v_d = v_d * 0.2  # scale velocity

        P = np.eye(2 + 1)  # Include slack variable
        q_vector = np.zeros(2 + 1)

        A_eq = np.hstack([J, np.zeros((2, 1))])  # Include slack
        b_eq = v_d

        G = np.vstack([np.eye(2 + 1), -np.eye(2 + 1)])
        h = np.hstack([self.params['q_max'], 10, -np.array(self.params['q_min']), 0])

        # Avoid obstacle
        A_obs, b_obs, v_d = self.avoid_obstacle(self.q, v_d)
        if A_obs is not None:
            G = np.vstack([G, np.hstack([A_obs, [0]])])
            h = np.hstack([h, b_obs])

        q_dot_opt = solve_qp(P, q_vector, G, h, A_eq, b_eq, solver='cvxopt')

        if q_dot_opt is not None:
            self.q = self.q + q_dot_opt[:2] * 0.1
            self.q = np.mod(self.q, 2 * np.pi)  # keep angles within [0, 2π]

        pos = self.forward_kinematics(self.q)
        self.line.set_data([0, self.params['l1'] * np.cos(self.q[0]),
                            self.params['l1'] * np.cos(self.q[0]) + self.params['l2'] * np.cos(self.q[0] + self.q[1])],
                           [0, self.params['l1'] * np.sin(self.q[0]),
                            self.params['l1'] * np.sin(self.q[0]) + self.params['l2'] * np.sin(self.q[0] + self.q[1])])
        self.end_effector.set_data([pos[0]], [pos[1]])

        return self.line, self.end_effector

    def animate(self):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_aspect('equal')

        self.line, = ax.plot([], [], 'bo-', lw=2, markersize=10, label='Robot Links')
        self.end_effector, = ax.plot([], [], 'ro', markersize=10, label='End-Effector')
        ax.plot(self.obstacle_position[0], self.obstacle_position[1], 'ko', markersize=15, label='Obstacle')

        ax.set_xlabel('X position')
        ax.set_ylabel('Y position')
        ax.set_title('2-DOF Robot Arm Animation')
        ax.legend()
        ax.grid(True)

        ani = animation.FuncAnimation(fig, self.update, frames=range(0, 100), interval=50, blit=True)
        plt.show()

# Parameters for the 2-DOF robot arm
params = {
    'l1': 1.5,
    'l2': 1.0,
    'q_min': [-1.0, -1.0],
    'q_max': [1.0, 1.0],
    'target_position': np.array([1, 1])
}

robot_arm = ARM(params)
robot_arm.animate()
