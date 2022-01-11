import numpy as np
import scipy.linalg
import modern_robotics as mr
import csv
np.set_printoptions(precision=3, suppress=True)

# Using a class in order to create easy unit tests
class RobotController:

    def __init__(self, filename, length=0.235, width=0.15, radius=0.0475):
        self.filename  = filename
        self.csvfile   = open(filename, "w")
        self.csvwriter = csv.writer(self.csvfile)

        # Robot Dimensions for Odometry
        self.length = length
        self.width  = width
        self.radius = radius

        # Step counter for trajectories
        self.step = 1


    ###################################################################################################################
    #================================================== Milestone 1 ==================================================#
    ###################################################################################################################

    def get_F_matrix(self):
        temp = 1 / (self.length + self.width)
        F = (self.radius / 4) * np.array([[-1 * temp, temp, temp, -1 * temp],
                                          [1, 1, 1, 1],
                                          [-1, 1, -1, 1]])
        return F

    def next_state(self, current_config, controls, timestep, max_arm_omega, max_wheel_omega, gripper_state):
        """
        Calculates the next state of the robot using odometry and a simple Euler step

        :param current_config:  A 12-vector representing the current configuration of the
                                robot (3 variables for the chassis configuration, 5 variables
                                for the arm configuration, and 4 variables for the wheels)
        :param controls:        A 9-vector of controls indicating the arm and wheel speeds
                                (5 variables for arm joint and 4 for the wheel speeds)
        :param timestep:        A timestep delta t
        :param max_arm_omega:   A positive real value indicating the maximum angular speed
                                of the arms.
        :param max_wheel_omega: A positive real value indicating the maximum angular speed
                                of the wheels
        :param gripper_state:   A boolean value 0 or 1 indicating whether the gripper is
                                open (0) or closed (1)
        :return:                A 12-vector representing the configuration of the robot time
                                at the next timestep.
        """
        # Convert to np arrays
        current_config = np.array(current_config)
        controls = np.array(controls)

        # Ensure that the controls are within bounds
        arm_controls = controls[0:5]
        arm_controls[arm_controls > max_arm_omega] = max_arm_omega
        arm_controls[arm_controls < -1 * max_arm_omega] = -1 * max_arm_omega

        wheel_controls = controls[5:9]
        wheel_controls[wheel_controls > max_wheel_omega] = max_wheel_omega
        wheel_controls[wheel_controls < -1 * max_wheel_omega] = -1 * max_wheel_omega

        # Find the new arm joint angles:
        current_arm_config = current_config[3:8]
        d_arm_config = arm_controls * timestep
        final_arm_config = current_arm_config + d_arm_config

        # Find the new wheel angles
        current_wheel_config = current_config[8:12]
        d_wheel_config = wheel_controls * timestep
        final_wheel_config = current_wheel_config + d_wheel_config

        # Use odometry to get the new chassis configuration
        current_chassis_config = current_config[0:3]
        # Using equation 13.34 on page 569 of the book to get the twist V_b
        F = self.get_F_matrix()
        V_b = F @ d_wheel_config

        # Use equation 13.35 to get the change in chassis configuration d_q_b
        if V_b[0] == 0:
            d_q_b = [0, V_b[1], V_b[2]]
        else:
            d_phi_b = V_b[0]
            d_x_b   = (V_b[1] * np.sin(V_b[0]) + V_b[2] * (np.cos(V_b[0]) - 1)) / V_b[0]
            d_y_b   = (V_b[2] * np.sin(V_b[0]) + V_b[1] * (1 - np.cos(V_b[0]))) / V_b[0]
            d_q_b = [d_phi_b, d_x_b, d_y_b]
        final_chassis_config = current_chassis_config + d_q_b

        # Write to CSV as 13-vector with the gripper state and return the 12-vector config for the next iteration
        next_config = np.concatenate((final_chassis_config, final_arm_config, final_wheel_config))
        final_config = np.append(next_config, gripper_state)
        self.csvwriter.writerow(final_config)
        return next_config


    ###################################################################################################################
    #================================================== Milestone 2 ==================================================#
    ###################################################################################################################


    @staticmethod
    def get_trajectory_time(X_start, X_end, linear_vel=0.25, angular_vel=0.25):
        """
        Calculates the time Tf for the trajectory to take by finding the distance between the
        two configurations

        :param X_start:     The start configuration of the trajectory
        :param X_end:       The end configuration of the trajectory
        :param linear_vel:  The maximum linear velocity for the end effector
        :param angular_vel: The maximum angular velocity for the end effector
        :return:            The time Tf that the trajectory should take along the path.
        """

        R_start, p_start = mr.TransToRp(X_start)
        R_end, p_end = mr.TransToRp(X_end)

        # Calculate the linear distance and time using simple Euclidean distance
        linear_distance = (p_end[0] - p_start[0]) ** 2 + \
                          (p_end[1] - p_start[1]) ** 2 + \
                          (p_end[2] - p_start[2]) ** 2
        linear_distance = np.sqrt(linear_distance)
        linear_time = linear_distance / linear_vel

        # Get the angular distance and time by calculating the angle of both configurations and subtracting
        angular_distance = np.arccos((np.trace(R_end) - 1) / 2) - np.arccos((np.trace(R_start) - 1) / 2)
        angular_time  = angular_distance / angular_vel

        if linear_time < angular_time:
            return angular_time
        else:
            return linear_time

    def parse_transformation_matrix_to_csv(self, T, gripper_state):
        out = [T[0][0], T[0][1], T[0][2], T[1][0], T[1][1], T[1][2], T[2][0], T[2][1], T[2][2],
               T[0][3], T[1][3], T[2][3], gripper_state]
        self.csvwriter.writerow(out)

    def create_trajectory_step(self, X_start, X_end, k, gripper_state):
        self.step += 1
        Tf = self.get_trajectory_time(X_start, X_end)

        if Tf == 0:
            trajectory_step = [X_end for _ in range(0, int((2 * k) / 0.01))]
        else:
            trajectory_step = mr.ScrewTrajectory(X_start, X_end, Tf, (Tf * k) / 0.01, 5)

        for config in trajectory_step:
            self.parse_transformation_matrix_to_csv(config, gripper_state)

        return trajectory_step

    def trajectory_generator(self, T_se_initial, T_sc_initial, T_sc_final, T_ce_grasp, T_ce_standoff, k):
        """
        Generates the reference trajectory for the end-effector frame {e}.
        This consists of 8 concatenated trajectory segments, with each beginning and ending at rest.

        These segments are:
        1. A trajectory to move the gripper from an initial configuration to a "standoff" configuration
        2. A trajectory to move the gripper from the "standoff" configuration to the grasp configuration
        3. Closing the gripper
        4. A trajectory to move the gripper back to the "standoff" configuration
        5. A trajectory to move the gripper to a "standoff" configuration above the final configuration
        6. A trajectory to move the gripper to the final configuration of the object
        7. Opening the gripper
        8. A trajectory to move the gripper back to the "standoff" configuration

        :param T_se_initial:  The initial configuration of the end-effector in the reference trajectory
        :param T_sc_initial:  The cube's initial configuration
        :param T_sc_final:    The cube's final configuration
        :param T_ce_grasp:    The end-effector's configuration relative to the cube when it's grasping the cube
        :param T_ce_standoff: The end-effector's standoff configuration above the cube, before and after grasping
                              relative to the cube. This specifies the configuration before lowering to T_ce_grasp
        :param k:             The number of trajectory reference configurations per 0.01 seconds

        :return:              A representation of the N configurations of the end-effector along the entire trajectory.
                              Each reference point represents a transformation matrix T_se of the end effector frame {e}
                              relative to {s} at an instant in time, plus the gripper state (0 or 1).
                              For example, if the trajectory takes 30 seconds, then there you will have ~30k/0.01
                              reference configurations depending on how boundary conditions are treated. This output
                              will follow the following schema:
                              r11, r12, r13, r21, r22, r23, r31, r32, r33, px, py ,pz, gripper state
                              with the first 12 variables being decided by the transformation matrix T_se at that time.
        """

        # Step 1: Move from T_se_initial to T_se_standoff_initial
        # This can be calculated by multiplying T_sc_initial @ T_ce_standoff = T_se_standoff_initial
        T_se_standoff_initial = T_sc_initial @ T_ce_standoff
        s1 = self.create_trajectory_step(T_se_initial, T_se_standoff_initial, k, 0)

        # Step 2: Move from T_se_standoff_initial to T_se_grasp_initial
        # This can be calculated by multiplying T_sc_initial and T_ce_grasp
        T_se_grasp_initial = T_sc_initial @ T_ce_grasp
        s2 = self.create_trajectory_step(T_se_standoff_initial, T_se_grasp_initial, k, 0)

        # Step 3: Close gripper for 1 second
        s3 = self.create_trajectory_step(T_se_grasp_initial, T_se_grasp_initial, k, 1)

        # Step 4: Move back to standoff configuration
        s4 = self.create_trajectory_step(T_se_grasp_initial, T_se_standoff_initial, k, 1)

        # Step 5: Move from T_se_standoff_initial to T_se_standoff_final
        T_se_standoff_final = T_sc_final @ T_ce_standoff
        s5 = self.create_trajectory_step(T_se_standoff_initial, T_se_standoff_final, k, 1)

        # Step 6: Move from T_se_standoff_final to T_se_grasp_final
        T_se_grasp_final = T_sc_final @ T_ce_grasp
        s6 = self.create_trajectory_step(T_se_standoff_final, T_se_grasp_final, k, 1)

        # Step 7: Open gripper for 1 second
        s7 = self.create_trajectory_step(T_se_grasp_final, T_se_grasp_final, k, 0)

        # Step 8: Move back to the T_se_standoff_final
        s8 = self.create_trajectory_step(T_se_grasp_final, T_se_standoff_final, k, 0)

        full_trajectory = np.concatenate((s1, s2, s3, s4, s5, s6, s7, s8))

        return full_trajectory

    ###################################################################################################################
    #================================================== Milestone 3 ==================================================#
    ###################################################################################################################

    @staticmethod
    def find_T_0e(M_0e, Blist, config):
        # Find T_0e through the forward kinematics of the arm
        thetalist = np.array(config[3:9])
        T_0e = mr.FKinBody(M_0e, Blist, thetalist)
        return T_0e

    def find_T_se(self, M_0e, Blist, config, T_b0):
        T_0e = self.find_T_0e(M_0e, Blist, config)
        phi = config[0]
        x   = config[1]
        y   = config[2]
        T_sb = np.array([[np.cos(phi), -1*np.sin(phi), 0,      x],
                         [np.sin(phi),    np.cos(phi), 0,      y],
                         [          0,              0, 1, 0.0963],
                         [          0,              0, 1,      0]])
        return (T_sb @ T_b0) @ T_0e

    def get_jacobian(self, M_0e, Blist, config, T_b0):
        """
        Finds the Jacobian matrix for the robot depending on the configuration

        :param M_0e:   The home configuration of the robot
        :param Blist:  The joint screw axes in the end-effector frame when the manipulator is at home
        :param config: The configuration (phi, x, y, theta_1, theta_2, ..., theta_5)
        :param T_b0:   Transformation matrix from the {b} frame to the {0} frame
        :return:       The Jacobian matrix for the base of the robot
        """
        # Find T_0e through the forward kinematics of the arm
        thetalist = config[3:9]
        T_0e = self.find_T_0e(M_0e, Blist, config)

        # Use T_0e, T_b0, and the F matrix to get the base Jacobian
        F = self.get_F_matrix()
        m_zero = np.zeros(len(F[0]))
        F6 = np.array((m_zero, m_zero, F[0], F[1], F[2], m_zero))
        J_base = mr.Adjoint(mr.TransInv(T_0e) @ mr.TransInv(T_b0)) @ F6

        # Find the body Jacobian through the library method
        J_body = mr.JacobianBody(Blist, thetalist)

        J = np.concatenate((J_base, J_body), axis=1)
        return J

    @staticmethod
    def feedback_command_ff_pi(T_se, T_se_d, T_se_d_next, K_p, K_i, timestep):
        """
        Calculates the end-effector twist V necessary to follow the trajectory defined by T_se and T_se_d.
        This uses the feedforward, proportional, and integral terms for the controller.

        :param T_se:        The current actual end-effector configuration.
        :param T_se_d:      The current end-effector reference configuration
        :param T_se_d_next: The end-effector reference configuration at the next timestep
        :param K_p:         The PI gain matrix for proportional control
        :param K_i:         The PI gain matrix for integral control
        :param timestep:    The timestep between the reference configurations
        :return:            The commanded end-effector twist V expressed in the end-effector frame {e}
        """

        # Feed forward term
        Vd      = (1 / timestep) * np.array(mr.MatrixLog6(mr.TransInv(T_se_d) @ T_se_d_next))
        Vd      = mr.se3ToVec(Vd)
        ff_term = mr.Adjoint(mr.TransInv(T_se) @ T_se_d) @ Vd

        # Proportional and integral terms
        X_err             = mr.MatrixLog6(mr.TransInv(T_se) @ T_se_d)
        X_err             = mr.se3ToVec(X_err)
        proportional_term = np.dot(K_p, X_err)
        integral_term     = np.dot(K_i, (X_err * timestep))

        return ff_term + proportional_term + integral_term

    @staticmethod
    def feedback_command_pi(T_se, T_se_d, K_p, K_i, timestep):
        """
        Calculates the end-effector twist V necessary to follow the trajectory defined by T_se and T_se_d.
        This uses the feedforward and integral terms for the controller.

        :param T_se:        The current actual end-effector configuration.
        :param T_se_d:      The current end-effector reference configuration
        :param K_p:         The PI gain matrix for proportional control
        :param K_i:         The PI gain matrix for integral control
        :param timestep:    The timestep between the reference configurations
        :return:            The commanded end-effector twist V expressed in the end-effector frame {e}
        """

        # Proportional and integral terms
        X_err = mr.MatrixLog6(mr.TransInv(T_se) @ T_se_d)
        X_err = mr.se3ToVec(X_err)
        proportional_term = np.dot(K_p, X_err)
        integral_term = np.dot(K_i, (X_err * timestep))

        return proportional_term + integral_term

    @staticmethod
    def feedback_command_ff_p(T_se, T_se_d, T_se_d_next, K_p, timestep):
        """
        Calculates the end-effector twist V necessary to follow the trajectory defined by T_se and T_se_d.
        This uses the feedforward and proportional terms for the controller.


        :param T_se:        The current actual end-effector configuration.
        :param T_se_d:      The current end-effector reference configuration
        :param T_se_d_next: The end-effector reference configuration at the next timestep
        :param K_p:         The PI gain matrix for proportional control
        :param timestep:    The timestep between the reference configurations
        :return:            The commanded end-effector twist V expressed in the end-effector frame {e}
        """

        # Feed forward term
        Vd = (1 / timestep) * np.array(mr.MatrixLog6(mr.TransInv(T_se_d) @ T_se_d_next))
        Vd = mr.se3ToVec(Vd)
        ff_term = mr.Adjoint(mr.TransInv(T_se) @ T_se_d) @ Vd

        # Proportional term
        X_err = mr.MatrixLog6(mr.TransInv(T_se) @ T_se_d)
        X_err = mr.se3ToVec(X_err)
        proportional_term = np.dot(K_p, X_err)

        return ff_term + proportional_term

    def feedback_control(self, T_se, T_se_d, T_se_d_next, K_p, K_i, timestep, M_0e, Blist, config, T_b0, is_printing):
        """
        Calculates the next configuration necessary to follow the trajectory delineated by T_se and T_se_d.

        :param T_se:        The current actual end-effector configuration.
        :param T_se_d:      The current end-effector reference configuration
        :param T_se_d_next: The end-effector reference configuration at the next timestep
        :param K_p:         The PI gain matrix for proportional control
        :param K_i:         The PI gain matrix for integral control
        :param timestep:    The timestep between the reference configurations
        :param M_0e:        The home configuration of the robot
        :param Blist:       The joint screw axes in the end-effector frame when the manipulator is at home
        :param config:      The configuration (phi, x, y, theta_1, theta_2, ..., theta_5)
        :param T_b0:        Transformation matrix from the {b} frame to the {0} frame
        :param is_printing: A boolean denoting whether the program should print output
        :return:            A configuration (u, theta dot) that consists of the wheel and arm angles.
                            This can be used in the `next_state` function
        """
        V = self.feedback_command_ff_pi(T_se, T_se_d, T_se_d_next, K_p, K_i, timestep)
        J = self.get_jacobian(M_0e, Blist, config, T_b0)
        Jpinv = scipy.linalg.pinv(J, 0.00001)
        if is_printing: print(f"J:\n{J}\n\nV:\n{V}\n(u, theta):\n{Jpinv @ V}")
        return Jpinv @ V

    # In order to avoid any file writing errors after run we need to close the file
    def end(self):
        self.csvfile.close()

if __name__ == '__main__':
    # Fixed offset from the chassis frame {b} to the base frame of the arm {0}
    t_b0 = [[1, 0, 0, 0.1662],
            [0, 1, 0, 0],
            [0, 0, 1, 0.0026],
            [0, 0, 0, 1]]

    # Arm at home configuration (all joint angles 0) end effector {e} relative to {0}
    m_0e = [[1, 0, 0, 0.033],
            [0, 1, 0, 0],
            [0, 0, 1, 0.6546],
            [0, 0, 0, 1]]

    # Initial and final configuration of the cube
    t_sc_initial = [[1, 0, 0, 1],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0.025],
                    [0, 0, 0, 1]]
    t_sc_final   = [[0, 1, 0, 0],
                    [-1, 0, 0, -1],
                    [0, 0, 1, 0.025],
                    [0, 0, 0, 1]]

    # Blist
    blist = np.array([[0, 0, 1, 0, 0.033, 0],
                      [0, -1, 0, -0.5076, 0, 0],
                      [0, -1, 0, -0.3526, 0, 0],
                      [0, -1, 0, -0.2176, 0, 0],
                      [0, 0, 1, 0, 0, 0]]).T

    controller = RobotController("../results/best/output1.csv")

