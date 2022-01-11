from unittest import TestCase
from main import RobotController
import numpy as np
np.set_printoptions(suppress=True)

class TestRobotController(TestCase):
    def test_next_state(self):
        controller = RobotController("../results/tests/next-state/test1.csv")
        configuration = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # Set with robot initial config
        controls = [0, 0, 0, 0, 0, 10, 10, 10, 10]
        for i in range(1, 101):
            configuration = controller.next_state(configuration, controls,
                                                  0.01, 10, 10, 0)
            assert (len(configuration) == 12)
        controller.end()

        controller = RobotController("../results/tests/next-state/test2.csv")
        configuration = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # Set with robot initial config
        controls = [0, 0, 0, 0, 0, -100, 100, -100, 100]
        for i in range(1, 101):
            configuration = controller.next_state(configuration, controls,
                                                  0.01, 10, 10, 0)
        controller.end()

        controller = RobotController("../results/tests/next-state/test3.csv")
        configuration = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # Set with robot initial config
        controls = [0, 0, 0, 0, 0, -10, 10, 10, -10]
        for i in range(1, 101):
            configuration = controller.next_state(configuration, controls,
                                                  0.01, 5, 5, 0)
        controller.end()

    def test_get_trajectory_time(self):
        return

    def test_trajectory_generator(self):
        # Fixed offset from the chassis frame {b} to the base frame of the arm {0}
        T_b0 = np.array([[1, 0, 0, 0.1662],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0.0026],
                         [0, 0, 0, 1]])

        # Arm at home configuration (all joint angles 0) end effector {e} relative to {0}
        M_0e = np.array([[1, 0, 0, 0.033],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0.6546],
                         [0, 0, 0, 1]])

        T_se_initial = T_b0 @ M_0e

        T_sc_initial = np.array([[1, 0, 0, 1],
                                 [0, 1, 0, 0],
                                 [0, 0, 1, 0.025],
                                 [0, 0, 0, 1]])

        T_sc_final = np.array([[0, 1, 0, 0],
                               [-1, 0, 0, -1],
                               [0, 0, 1, 0.025],
                               [0, 0, 0, 1]])

        T_ce_grasp = np.array([[0, 0, 1, 0],
                               [0, 1, 0, 0],
                               [-1, 0, 0, 0],
                               [0, 0, 0, 1]])

        T_ce_standoff = np.array([[0, 0, 1, 0],
                                  [0, 1, 0, 0],
                                  [-1, 0, 0, 0.25],
                                  [0, 0, 0, 1]])

        controller = RobotController("../results/tests/trajectory-generator/test1.csv")
        traj = controller.trajectory_generator(T_se_initial,
                                               T_sc_initial,
                                               T_sc_final,
                                               T_ce_grasp,
                                               T_ce_standoff,
                                               1)
        controller.end()

    def test_feedback_control(self):
        controller = RobotController("../results/tests/feedback-control/test1.csv")
        config  = [0, 0, 0, 0, 0, 0.2, -1.6, 0]
        Xd      = np.array([[ 0, 0, 1, 0.5],
                            [ 0, 1, 0,   0],
                            [-1, 0, 0, 0.5],
                            [ 0, 0, 0,   1]])

        Xd_next = np.array([[ 0, 0, 1, 0.6],
                            [ 0, 1, 0,   0],
                            [-1, 0, 0, 0.3],
                            [0, 0, 0,    1]])

        X       = np.array([[ 0.170, 0, 0.985, 0.387],
                            [     0, 1,     0,     0],
                            [-0.985, 0, 0.170, 0.570],
                            [     0, 0,     0,     1]])

        # Fixed offset from the chassis frame {b} to the base frame of the arm {0}
        T_b0 = np.array([[1, 0, 0, 0.1662],
                         [0, 1, 0,      0],
                         [0, 0, 1, 0.0026],
                         [0, 0, 0,      1]])

        # Arm at home configuration (all joint angles 0) end effector {e} relative to {0}
        M_0e = np.array([[1, 0, 0,  0.033],
                         [0, 1, 0,      0],
                         [0, 0, 1, 0.6546],
                         [0, 0, 0,      1]])

        Blist = np.array([[0,  0, 1,       0, 0.033, 0],
                          [0, -1, 0, -0.5076,     0, 0],
                          [0, -1, 0, -0.3526,     0, 0],
                          [0, -1, 0, -0.2176,     0, 0],
                          [0,  0, 1,       0,     0, 0]]).T

        q  = controller.feedback_control(X, Xd, Xd_next, np.eye(6), np.eye(6), 0.01, M_0e, Blist, config, T_b0, False)

        controller.end()
