from math import pi

import matplotlib.pyplot as plt
import numpy as np
from Helper.rotplot import rotplot
from matplotlib.animation import FuncAnimation
from scipy import io
from scipy.spatial.transform import Rotation as R


def plot_orientation_graph(ts, gyro_rpy, acc_rpy, compli_rpy, vicon_rpy, mad_rpy,ukf_rpy):
    plt.figure()
    plt.subplot(311)
    plt.plot(ts, gyro_rpy[0, :], label="gyro")
    plt.plot(ts, acc_rpy[0, :], label="acc")
    plt.plot(ts, compli_rpy[0, :], label="compli")
    plt.plot(ts, vicon_rpy[0, :], label="vicon")
    plt.plot(ts, mad_rpy[0, :], label="madgwick")
    plt.plot( ts,ukf_rpy[0, :], label="ukf")
    plt.xlabel("Time")
    plt.ylabel("Roll")
    plt.legend()
    plt.title("Roll(X-axis)")

    plt.subplot(312)
    plt.plot(ts, gyro_rpy[1, :], label="gyro")
    plt.plot(ts, acc_rpy[1, :], label="acc")
    plt.plot(ts, compli_rpy[1, :], label="compli")
    plt.plot(ts, vicon_rpy[1, :], label="vicon")
    plt.plot(ts, mad_rpy[1, :], label="madgwick")
    plt.plot(ts, ukf_rpy[1, :], label="ukf")
    plt.xlabel("Time")
    plt.ylabel("Pitch")
    plt.legend()
    plt.title("Pitch(Y-axis)")

    plt.subplot(313)
    plt.plot(ts, gyro_rpy[2, :], label="gyro")
    plt.plot(ts, acc_rpy[2, :], label="acc")
    plt.plot(ts, compli_rpy[2, :], label="compli")
    plt.plot(ts, vicon_rpy[2, :], label="vicon")
    plt.plot(ts, mad_rpy[2, :], label="madgwick")
    plt.plot(ts, ukf_rpy[2, :], label="ukf")
    plt.xlabel("Time")
    plt.ylabel("Yaw")
    plt.legend()
    plt.title("Yaw(Z-axis)")
    plt.show()


def plot_orientation_graph_test(ts, gyro_rpy, acc_rpy, compli_rpy, mad_rpy,ukf_rpy):
    plt.figure()
    plt.subplot(311)
    plt.plot(ts, gyro_rpy[0, :], label="gyro")
    plt.plot(ts, acc_rpy[0, :], label="acc")
    plt.plot(ts, compli_rpy[0, :], label="compli")
    plt.plot(ts, mad_rpy[0, :], label="madgwick")
    plt.plot(ts, ukf_rpy[0, :], label="ukf")
    plt.xlabel("Time")
    plt.ylabel("Roll")
    plt.legend()
    plt.title("Roll(X-axis)")

    plt.subplot(312)
    plt.plot(ts, gyro_rpy[1, :], label="gyro")
    plt.plot(ts, acc_rpy[1, :], label="acc")
    plt.plot(ts, compli_rpy[1, :], label="compli")
    plt.plot(ts, mad_rpy[1, :], label="madgwick")
    plt.plot(ts, ukf_rpy[1, :], label="ukf")
    plt.xlabel("Time")
    plt.ylabel("Pitch")
    plt.legend()
    plt.title("Pitch(Y-axis)")

    plt.subplot(313)
    plt.plot(ts, gyro_rpy[2, :], label="gyro")
    plt.plot(ts, acc_rpy[2, :], label="acc")
    plt.plot(ts, compli_rpy[2, :], label="compli")
    plt.plot(ts, mad_rpy[2, :], label="madgwick")
    plt.plot(ts, ukf_rpy[2, :], label="ukf")
    plt.xlabel("Time")
    plt.ylabel("Yaw")
    plt.legend()
    plt.title("Yaw(Z-axis)")
    plt.show()


def update_graph(
    num, gyro_rot, acc_rot, compli_rot, mad_rot,ukf_rot, vicon_rot, ax, ax1, ax2, ax3, ax4,ax5
):
    ax.clear()
    ax1.clear()
    ax2.clear()
    ax3.clear()
    ax4.clear()
    ax5.clear()
    rotplot(gyro_rot[:, :, num], ax, "Gyroscope")
    rotplot(acc_rot[:, :, num], ax1, "Accelerometer")
    rotplot(compli_rot[:, :, num], ax2, "Complimentary Filter")
    rotplot(mad_rot[:, :, num], ax3, "Madgwick Filter")
    rotplot(ukf_rot[:, :, num], ax4, "UKF")
    rotplot(vicon_rot[:, :, num], ax5, "Vicon")
    if num == len(gyro_rot[0, 0, :]) - 1:
        plt.close()


def plot_orintaion_3d(gyro_rot, acc_rot, compli_rot, vicon_rot, mad_rot,ukf_rot):
    fig = plt.figure()
    ax = fig.add_subplot(161, projection="3d")
    ax1 = fig.add_subplot(162, projection="3d")
    ax2 = fig.add_subplot(163, projection="3d")
    ax3 = fig.add_subplot(164, projection="3d")
    ax4 = fig.add_subplot(165, projection="3d")
    ax5 = fig.add_subplot(166, projection="3d")
    ani = FuncAnimation(
        fig,
        update_graph,
        fargs=(
            gyro_rot,
            acc_rot,
            compli_rot,
            mad_rot,
            ukf_rot,
            vicon_rot,
            ax,
            ax1,
            ax2,
            ax3,
            ax4,
            ax5,
        ),
        frames=len(gyro_rot[0, 0, :]),
        interval=10,
        repeat=False,
    )
    plt.show()
    ani.save("video2.mp4", writer="ffmpeg", fps=60, dpi=300)


def update_graph_test(num, gyro_rot, acc_rot, compli_rot, mad_rot,ukf_rot, ax, ax1, ax2, ax3,ax4):
    ax.clear()
    ax1.clear()
    ax2.clear()
    ax3.clear()
    ax4.clear()
    rotplot(gyro_rot[:, :, num], ax, "Gyroscope")
    rotplot(acc_rot[:, :, num], ax1, "Accelerometer")
    rotplot(compli_rot[:, :, num], ax2, "Complimentary Filter")
    rotplot(mad_rot[:, :, num], ax3, "Madgwick Filter")
    rotplot(ukf_rot[:, :, num], ax4, "UKF")
    if num == len(gyro_rot[0, 0, :]) - 1:
        plt.close()


def plot_orintaion_3d_test(gyro_rot, acc_rot, compli_rot, mad_rot,ukf_rot):
    fig = plt.figure()
    ax = fig.add_subplot(151, projection="3d")
    ax1 = fig.add_subplot(152, projection="3d")
    ax2 = fig.add_subplot(153, projection="3d")
    ax3 = fig.add_subplot(154, projection="3d")
    ax4 = fig.add_subplot(155, projection="3d")
    ani = FuncAnimation(
        fig,
        update_graph_test,
        fargs=(
            gyro_rot,
            acc_rot,
            compli_rot,
            mad_rot,
            ukf_rot,
            ax,
            ax1,
            ax2,
            ax3,
            ax4,
        ),
        frames=len(gyro_rot[0, 0, :]),
        interval=10,
        repeat=False,
    )
    plt.show()
    ani.save("video2.mp4", writer="ffmpeg", fps=60, dpi=300)


def rot_to_rpy(mat):
    r = np.arctan2(mat[2, 1], mat[2, 2])
    p = np.arctan2(-mat[2, 0], np.sqrt(mat[2, 1] ** 2 + mat[2, 2] ** 2))
    y = np.arctan2(mat[1, 0], mat[0, 0])
    return np.array([r, p, y])


def rpy_to_rot(rpy):
    r = rpy[0]
    p = rpy[1]
    y = rpy[2]
    rot = np.array(
        [
            [
                np.cos(y) * np.cos(p),
                np.cos(y) * np.sin(p) * np.sin(r) - np.sin(y) * np.cos(r),
                np.cos(y) * np.sin(p) * np.cos(r) + np.sin(y) * np.sin(r),
            ],
            [
                np.sin(y) * np.cos(p),
                np.sin(y) * np.sin(p) * np.sin(r) + np.cos(y) * np.cos(r),
                np.sin(y) * np.sin(p) * np.cos(r) - np.cos(y) * np.sin(r),
            ],
            [-np.sin(p), np.cos(p) * np.sin(r), np.cos(p) * np.cos(r)],
        ]
    )
    return rot
