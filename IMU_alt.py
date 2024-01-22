from math import pi

import matplotlib.pyplot as plt
import numpy as np
from Helper.rotplot import rotplot
from numpy.linalg import norm
from scipy import io
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from Helper.ukf import ukf_main


class IMU_alt:
    def __init__(self, imu_raw_loc, vicon_loc, imu_params_loc, is_test,file_number=1):
        imu_raw = io.loadmat(imu_raw_loc)

        # imuparams
        x = io.loadmat(imu_params_loc)
        self.imuparams = x["IMUParams"]

        # imu_raw
        imu_raw_val = imu_raw["vals"]
        self.imu_ts = imu_raw["ts"][0]

        # vicon
        if is_test == False:
            vicon = io.loadmat(vicon_loc)
            self.vicon_rot = vicon["rots"]
            self.vicon_ts = vicon["ts"][0]

        # gyro
        self.high_pass_alpha = 0.999
        gyro_vals_raw = imu_raw_val[3:6, :]

        gyro_vals_raw[[0, 1]] = gyro_vals_raw[[1, 0]]
        gyro_vals_raw[[1, 2]] = gyro_vals_raw[[2, 1]]

        bg = np.average(gyro_vals_raw[:, :200], axis=1)
        bg = np.reshape(bg, (3, 1))

        # actual gyro values
        self.gyro_vals = (gyro_vals_raw - bg) * 0.3 * (pi / 180) * (3300 / 1023)

        # times
        if is_test == False:
            self.get_start_and_end_time()
        else:
            self.start_time = self.imu_ts[0]
            self.start_index = 0
            self.end_time = self.imu_ts[-1]
            self.end_index = self.imu_ts.shape[0] - 1

        # initial gyro orientation
        if is_test == False:
            self.get_initial_gyro_orien()
            self.inital_gyro_orien = np.reshape(self.inital_gyro_orien, (3, 1))
        else:
            self.inital_gyro_orien = np.array([[0], [0], [0]])
        # acc
        self.gamma = 0.2
        acc_vals_raw = imu_raw_val[0:3, :]
        self.acc_vals = np.zeros((3, acc_vals_raw.shape[1]))
        self.acc_vals[0, :] = (
            acc_vals_raw[0, :] * self.imuparams[0, 0] + self.imuparams[1, 0]
        )
        self.acc_vals[1, :] = (
            acc_vals_raw[1, :] * self.imuparams[0, 1] + self.imuparams[1, 1]
        )
        self.acc_vals[2, :] = (
            acc_vals_raw[2, :] * self.imuparams[0, 2] + self.imuparams[1, 2]
        )

        # sync timestamps
        self.imu_ts_sync = self.imu_ts[self.start_index : self.end_index + 1]

        # sync vicon data
        if is_test == False:
            self.get_vicon_rot_sync()
            self.get_vicon_rpy_sync()

        # compli filter params
        self.alpha = np.array([0.8, 0.8, 0.25])

        # madgwick filter params
        self.beta = 0.1

        # ukf filter
        self.ukf_rpy=ukf_main(file_number,is_test)

    def get_gyro_orientation(self):
        self.gyro_rpy = np.zeros((3, self.gyro_vals.shape[1]))
        self.gyro_rpy[:, 0] = self.inital_gyro_orien[:, 0]
        self.gyro_rpy_un = np.zeros((3, self.gyro_vals.shape[1]))
        self.gyro_rpy_un[:, 0] = self.inital_gyro_orien[:, 0]
        self.gyro_rot = np.zeros((3, 3, self.gyro_vals.shape[1]))
        self.gyro_rot_un = np.zeros((3, 3, self.gyro_vals.shape[1]))
        self.gyro_val_corrected = np.zeros((3, self.gyro_vals.shape[1]))
        for i in range(1, self.gyro_vals.shape[1]):
            dt = self.imu_ts[i] - self.imu_ts[i - 1]
            Rot = R.from_euler("xyz", self.gyro_rpy[:, i - 1]).as_matrix()
            self.gyro_val_corrected[:, i] = np.matmul(Rot, self.gyro_vals[:, i])

            self.gyro_rpy_un[:, i] = (
                self.gyro_rpy_un[:, i - 1] + dt * self.gyro_val_corrected[:, i]
            )
            self.gyro_rpy[:, i] = self.high_pass_alpha * (
                self.gyro_rpy[:, i - 1]
                - self.gyro_rpy_un[:, i - 1]
                + self.gyro_rpy_un[:, i]
            )
            self.gyro_rot[:, :, i] = R.from_euler(
                "xyz", self.gyro_rpy[:, i]
            ).as_matrix()
            self.gyro_rot_un[:, :, i] = R.from_euler(
                "xyz", self.gyro_rpy_un[:, i]
            ).as_matrix()
        # sync gyro data
        self.gyro_rpy_sync = self.gyro_rpy[:, self.start_index : self.end_index + 1]
        self.gyro_rot_sync = self.gyro_rot[:, :, self.start_index : self.end_index + 1]
        self.gyro_rpy_un_sync = self.gyro_rpy_un[
            :, self.start_index : self.end_index + 1
        ]
        self.gyro_rot_un_sync = self.gyro_rot_un[
            :, :, self.start_index : self.end_index + 1
        ]

    def get_acc_orientation(self):
        self.acc_rpy = np.zeros((3, self.acc_vals.shape[1]))
        self.acc_rot = np.zeros((3, 3, self.acc_vals.shape[1]))
        for i in range(1, self.acc_vals.shape[1]):
            if i == 0:
                self.acc_rpy[:, i] = self.get_acc_rpy(self.acc_vals[:, i])
            else:
                self.acc_rpy[:, i] = (
                    self.gamma * self.get_acc_rpy(self.acc_vals[:, i])
                    + (1 - self.gamma) * self.acc_rpy[:, i - 1]
                )
            self.acc_rot[:, :, i] = R.from_euler("xyz", self.acc_rpy[:, i]).as_matrix()
            # self.acc_rpy[:, i] = (
            #     self.gamma * self.get_acc_rpy(self.acc_vals[:, i - 1])
            #     + (1 - self.gamma) * self.acc_rpy[:, i - 1]
            # )
            # self.acc_rot[:, :, i] = R.from_euler("xyz", self.acc_rpy[:, i]).as_matrix()

        # sync acc data
        self.acc_rpy_sync = self.acc_rpy[:, self.start_index : self.end_index + 1]
        self.acc_rot_sync = self.acc_rot[:, :, self.start_index : self.end_index + 1]

    def get_acc_rpy(self, acc_val):
        r = np.arctan2(acc_val[1], np.sqrt(acc_val[0] ** 2 + acc_val[2] ** 2))
        p = np.arctan2(-acc_val[0], np.sqrt(acc_val[1] ** 2 + acc_val[2] ** 2))
        y = np.arctan2(np.sqrt(acc_val[0] ** 2 + acc_val[1] ** 2), acc_val[2])
        return np.array([r, p, y])

    def compli_filter(self):
        self.compli_rpy = np.zeros((3, self.acc_rpy.shape[1]))
        self.compli_rot = np.zeros((3, 3, self.acc_rpy.shape[1]))
        for i in range(self.acc_rpy.shape[1]):
            self.compli_rpy[:, i] = (1 - self.alpha) * self.gyro_rpy[
                :, i
            ] + self.alpha * self.acc_rpy[:, i]
            self.compli_rot[:, :, i] = R.from_euler(
                "xyz", self.compli_rpy[:, i]
            ).as_matrix()
        # sync compli data
        self.compli_rpy_sync = self.compli_rpy[:, self.start_index : self.end_index + 1]
        self.compli_rot_sync = self.compli_rot[
            :, :, self.start_index : self.end_index + 1
        ]

    def reshape_vicon_rot(self):
        reshaped_vicon_rot = np.zeros((self.vicon_rot.shape[2], 3, 3))
        for i in range(self.vicon_rot.shape[2]):
            reshaped_vicon_rot[i, :, :] = self.vicon_rot[:, :, i]
        return reshaped_vicon_rot

    def get_start_and_end_time(self):
        if self.imu_ts[0] < self.vicon_ts[0]:
            for i in range(len(self.imu_ts)):
                if self.imu_ts[i] > self.vicon_ts[0]:
                    break
            self.start_time = self.imu_ts[i]
            self.start_index = i
        else:
            self.start_time = self.imu_ts[0]
            self.start_index = 0
        if self.imu_ts[-1] > self.vicon_ts[-1]:
            for i in range(len(self.imu_ts)):
                if self.imu_ts[i] > self.vicon_ts[-1]:
                    break
            self.end_time = self.imu_ts[i - 1]
            self.end_index = i - 1
        else:
            self.end_time = self.imu_ts[-1]
            self.end_index = len(self.imu_ts) - 1

    def get_vicon_rot_sync(self):
        self.vicon_rot_sync = np.zeros((3, 3, self.imu_ts_sync.shape[0]))
        for i in range(len(self.imu_ts_sync)):
            closest_vicon_ts = None
            clostest_vicon_index = 0
            for j in range(len(self.vicon_ts)):
                if (
                    closest_vicon_ts is None
                    or abs(self.vicon_ts[j] - self.imu_ts_sync[i]) < closest_vicon_ts
                ):
                    clostest_vicon_index = j
                    closest_vicon_ts = abs(self.vicon_ts[j] - self.imu_ts_sync[i])
            self.vicon_rot_sync[:, :, i] = self.vicon_rot[:, :, clostest_vicon_index]

    def get_vicon_rpy_sync(self):
        self.vicon_rpy_sync = np.zeros([3, len(self.imu_ts_sync)])
        for i in range(len(self.imu_ts_sync)):
            self.vicon_rpy_sync[:, i] = R.from_matrix(
                self.vicon_rot_sync[:, :, i]
            ).as_euler("xyz", degrees=False)

    def get_initial_gyro_orien(self):
        start_time = self.imu_ts[0]
        closest_vicon_ts = None
        closest_vicon_index = 0
        for i in range(len(self.vicon_ts)):
            if (
                closest_vicon_ts is None
                or abs(self.vicon_ts[i] - start_time) < closest_vicon_ts
            ):
                closest_vicon_index = i
                closest_vicon_ts = abs(self.vicon_ts[i] - start_time)
        self.inital_gyro_orien = R.from_matrix(
            self.vicon_rot[:, :, closest_vicon_index]
        ).as_euler("xyz", degrees=False)

    def madgwick_filter(self):
        mad_quat = np.zeros((4, self.gyro_vals.shape[1]))
        inter = R.from_euler("xyz", self.inital_gyro_orien[:, 0]).as_matrix()
        inter = inter.T
        mad_quat[:, 0] = R.from_matrix(inter).as_quat()
        mad_quat[:, 0] = mad_quat[:, 0] / norm(mad_quat[:, 0])
        mad_rpy = np.zeros((3, self.gyro_vals.shape[1]))
        mad_rpy[:, 0] = R.from_quat(mad_quat[:, 0]).as_euler("xyz", degrees=False)
        mad_rot = np.zeros((3, 3, self.gyro_vals.shape[1]))
        mad_rot[:, :, 0] = R.from_euler("xyz", mad_rpy[:, 0]).as_matrix()
        for i in range(1, len(self.imu_ts)):
            acc_norm = self.acc_vals[:, i] / norm(self.acc_vals[:, i])

            dt = self.imu_ts[i] - self.imu_ts[i - 1]
            w = np.array(
                [
                    [0],
                    [self.gyro_vals[0, i]],
                    [self.gyro_vals[1, i]],
                    [self.gyro_vals[2, i]],
                ],
            )
            gyro_quat_change = 0.5 * self.multiply_quaternions(mad_quat[:, i - 1], w)
            J = np.array(
                [
                    [
                        -2 * mad_quat[2, i - 1],
                        2 * mad_quat[3, i - 1],
                        -2 * mad_quat[0, i - 1],
                        2 * mad_quat[1, i - 1],
                    ],
                    [
                        2 * mad_quat[1, i - 1],
                        2 * mad_quat[0, i - 1],
                        2 * mad_quat[3, i - 1],
                        2 * mad_quat[2, i - 1],
                    ],
                    [0, -4 * mad_quat[1, i - 1], -4 * mad_quat[2, i - 1], 0],
                ]
            )
            f = np.array(
                [
                    [
                        2
                        * (
                            mad_quat[1, i - 1] * mad_quat[3, i - 1]
                            - mad_quat[0, i - 1] * mad_quat[2, i - 1]
                        )
                        - acc_norm[0],
                    ],
                    [
                        2
                        * (
                            mad_quat[0, i - 1] * mad_quat[1, i - 1]
                            + mad_quat[2, i - 1] * mad_quat[3, i - 1]
                        )
                        - acc_norm[1]
                    ],
                    [
                        2 * (0.5 - mad_quat[1, i - 1] ** 2 - mad_quat[2, i - 1] ** 2)
                        - acc_norm[2]
                    ],
                ]
            )
            del_f = J.T.dot(f)
            acc_change = -self.beta * (del_f / norm(del_f))
            combined_change = gyro_quat_change + acc_change
            combined_change = combined_change.reshape((4,))
            mad_quat[:, i] = mad_quat[:, i - 1] + dt * combined_change
            mad_quat[:, i] = mad_quat[:, i] / norm(mad_quat[:, i])
            inter = R.from_quat(mad_quat[:, i]).as_matrix()
            inter = inter.T
            mad_rpy[:, i] = R.from_matrix(inter).as_euler("zyx", degrees=False)
            mad_rpy[0, i] = -mad_rpy[0, i]
            mad_rot[:, :, i] = R.from_euler("xyz", mad_rpy[:, i]).as_matrix()
        # sync mad data
        self.mad_rpy_sync = mad_rpy[:, self.start_index : self.end_index + 1]
        self.mad_quat_sync = mad_quat[:, self.start_index : self.end_index + 1]
        self.mad_rot_sync = mad_rot[:, :, self.start_index : self.end_index + 1]

    def multiply_quaternions(self, q1, q2):
        w1, x1, y1, z1 = q1

        w2, x2, y2, z2 = q2
        result = np.array(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ]
        )
        return result
    
    def ukf_filter(self):
        self.ukf_rpy_sync = self.ukf_rpy[:, self.start_index : self.end_index + 1]
        self.ukf_rot = np.zeros((3, 3, len(self.imu_ts)))
        for i in range(len(self.imu_ts)):
            self.ukf_rot[:,:, i] = R.from_euler("xyz", self.ukf_rpy[:, i]).as_matrix()
        self.ukf_rot_sync = self.ukf_rot[:,:, self.start_index : self.end_index + 1]
