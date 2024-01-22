import numpy as np
from Helper.IMU import IMU
from Helper.IMU_alt import IMU_alt
from Helper.rotplot import rotplot
from Helper.utils import (
    plot_orientation_graph,
    plot_orientation_graph_test,
    plot_orintaion_3d,
    plot_orintaion_3d_test,
    rot_to_rpy,
    rpy_to_rot,
)
from Wrapper2 import ukf_main



def main():
    file_num = 1
    is_test = False
    imu_loc = '../Data/Train/IMU/imuRaw' + str(file_num) + '.mat'
    vicon_loc = '../Data/Train/Vicon/viconRot' + str(file_num) + '.mat'
    
    imu_params_loc = (
        "../IMUParams.mat"
    )
    try:
        imu = IMU(imu_loc, vicon_loc, imu_params_loc, is_test,file_num)
    except:
        imu = IMU_alt(imu_loc, vicon_loc, imu_params_loc, is_test,file_num)
    # calculate orientations
    imu.get_gyro_orientation()
    imu.get_acc_orientation()
    imu.compli_filter()
    imu.madgwick_filter()
    imu.ukf_filter()
    # plot roll pitch yaw
    if is_test == False:
        plot_orientation_graph(
            imu.imu_ts_sync,
            imu.gyro_rpy_sync,
            imu.acc_rpy_sync,
            imu.compli_rpy_sync,
            imu.vicon_rpy_sync,
            imu.mad_rpy_sync,
            imu.ukf_rpy_sync,
        )
    else:
        plot_orientation_graph_test(
            imu.imu_ts_sync,
            imu.gyro_rpy_sync,
            imu.acc_rpy_sync,
            imu.compli_rpy_sync,
            imu.mad_rpy_sync,
            imu.ukf_rpy_sync,
        )
    # plot 3d orientation
    if is_test == False:
        plot_orintaion_3d(
            imu.gyro_rot_sync,
            imu.acc_rot_sync,
            imu.compli_rot_sync,
            imu.vicon_rot_sync,
            imu.mad_rot_sync,
            imu.ukf_rot_sync,
        )

    else:
        plot_orintaion_3d_test(
            imu.gyro_rot_sync,
            imu.acc_rot_sync,
            imu.compli_rot_sync,
            imu.mad_rot_sync,
            imu.ukf_rot_sync,
        )


if __name__ == "__main__":
    main()
