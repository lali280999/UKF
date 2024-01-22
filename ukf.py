import numpy as np
from scipy import io
from scipy.linalg import cholesky
from scipy.spatial.transform import Rotation as R
from math import pi
import matplotlib.pyplot as plt
from scipy.spatial.transform import Slerp

def quatmult(q1,q2):
    qprod = np.zeros((4),dtype=float)
    qprod[0] = q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3]
    qprod[1] = q1[1]*q2[0] + q1[0]*q2[1] - q1[3]*q2[2] + q1[2]*q2[3]
    qprod[2] = q1[2]*q2[0] + q1[3]*q2[1] + q1[0]*q2[2] - q1[1]*q2[3]
    qprod[3] = q1[3]*q2[0] - q1[2]*q2[1] + q1[1]*q2[2] + q1[0]*q2[3]
    return qprod/np.linalg.norm(qprod)

def quatinv(q):
    return [q[0], -q[1], -q[2], -q[3]]

def quat2rot(q):
    
    sinalpha = np.linalg.norm(q[1:4])
    cosalpha = q[0]
    alpha = np.arctan2(sinalpha,cosalpha)
    if alpha != 0:
       # q = q/np.linalg.norm(q)
        e = q[1:4]/np.sin(alpha)
        e = e*alpha*2
    else:
        e = np.zeros((3), dtype=float)
    return e


def rot2quat(e):
    e = np.reshape(e,(3))
    alpha = np.linalg.norm(e)
    if alpha != 0:
        e = e/alpha
        q = np.asarray([np.cos(alpha/2),e[0]*np.sin(alpha/2),e[1]*np.sin(alpha/2),e[2]*np.sin(alpha/2)])
    else:
        q = np.asarray([1.0,0.0,0.0,0.0])

    return q

def weighted_sum(Y,weights):
    n = len(weights)
    sum = np.zeros((len(Y[:,0]),1),dtype=float)
    for i in range(n):
        sum = sum + np.reshape(weights[i]*Y[:,i],(len(Y[:,0]),1))
    return sum


def sigma_point(x_hat,Pkk,Q):

    #matrix square root of sum of covariance matrix and process noise covariance matrix
    S = cholesky(Pkk + Q, lower=False)

    n = len(S[0])

    W = np.zeros((6,2*n+1),dtype=float)

    for i in range(n):
        W[:,i+1] = np.sqrt(3)*S[:,i]
        W[:,i+n+1] = -np.sqrt(3)*S[:,i]

    Xi = np.zeros((7,2*n+1),dtype=float)

    qk_1 = x_hat[0:4,0]
    omega = x_hat[4:7,0]


    #Generating sigma points
    for i in range(2*n+1):
        qwi = rot2quat(W[0:3,i])
        q_tilde = quatmult(qk_1,qwi) 
        Xi[0:4,i] = q_tilde  
        Xi[4:7,i] = omega + W[3:6,i]

    return Xi


def propogate_sigma_points(Xi,n,dt):
    Y_pred = np.zeros([7,2*n+1],dtype=float)

    for i in range(2*n+1):
        mag = np.linalg.norm(Xi[4:7,i])
        if mag != 0:
            alpha_delta = mag*dt
            e_delta = Xi[4:7,i]/mag
            q_delta = np.asarray([np.cos(alpha_delta/2),e_delta[0]*np.sin(alpha_delta/2),e_delta[1]*np.sin(alpha_delta/2),e_delta[2]*np.sin(alpha_delta/2)])
            propagated_state = quatmult((Xi[0:4,i]),q_delta)
            Y_pred[0:4,i] = propagated_state
            Y_pred[4:7,i] = Xi[4:7,i]
        else:
            q_delta = np.asarray([1.0,0.0,0.0,0.0])
            propagated_state = quatmult((Xi[0:4,i]),q_delta)
            Y_pred[0:4,i] = propagated_state
            Y_pred[4:7,i] = Xi[4:7,i]

    return Y_pred
    


def calc_mean_covariance(Y,n,weights):
    x_hat_ = np.zeros((7,1),dtype=float)
    Adjustment_vectors = np.zeros((3,2*n+1))
    Pk_ = np.zeros((6,6),dtype=float)

    e_bar = [1.0,0.0,0.0]
    qt_bar = Y[0:4,0]

    iter = 0
    # Gradient descent to find the mean of the quaternion
    while np.linalg.norm(e_bar) > 1e-5 and iter < 1000:
        iter = iter + 1
        for i in range(2*n+1):
            ei = quatmult(Y[0:4,i],quatinv(qt_bar))
            Adjustment_vectors[:,i] = quat2rot(ei)

        # e_bar =np.sum(Adjustment_vectors,axis=1)/(2*n+1)
        e_bar = weighted_sum(Adjustment_vectors,weights)
        e_bar = np.ravel(e_bar)
        e_bar_quat = rot2quat(e_bar)
        qt_bar = quatmult(e_bar_quat,qt_bar)

    x_hat_[0:4] = np.reshape(qt_bar,(4,1))
    x_hat_[4:7] = weighted_sum(Y[4:7,:],weights) 

    for i in range(2*n+1):
        ei = quatmult(quatinv(qt_bar),Y[0:4,i])
        Adjustment_vectors[:,i] = quat2rot(ei)

    W_prime = np.zeros((6,2*n+1))

    for i in range(2*n+1):
        velocity_difference = Y[4:7,i] - x_hat_[4:7,0]
        rw = Adjustment_vectors[:,i]
        Wpi = np.concatenate((rw,velocity_difference),axis=0)
        Wpi = np.reshape(Wpi,(6,1))
        # W_prime = W_prime + np.matmul(Wpi,np.transpose(Wpi))
        Pk_ = Pk_ + weights[i]*np.matmul(Wpi,np.transpose(Wpi))
        W_prime[:,i] = np.ravel(Wpi)


    Pk_ = Pk_
    return x_hat_, Pk_, W_prime


def measurement_prediction(Yi_f,n):
    Z = np.zeros((6,2*n+1))

    g = np.asarray([0.0,0.0,0.0,1.0])

    for i in range(2*n+1):
        #get rotation matrix from quaternion
        # g_prime = np.matmul(rot_mat,g)
        #get gravity vector in world frame
        qk =Yi_f[0:4,i]
        g_prime = quatmult(quatmult(qk,g),quatinv(qk))
        # print("g_prime",g_prime)
        g_prime = quat2rot(g_prime)
        Z[0:3,i] = g_prime
        Z[3:6,i] = Yi_f[4:7,i]

    return Z

def measurement_mean_cov(Z_f,R_noise,n,weights):
    # Calculating the mean of the measurement sigma points
    zk_ = weighted_sum(Z_f,weights)
    zk_ = np.ravel(zk_)

    # Calculating the covariance of the measurement sigma points
    Pzz = np.zeros((6,6))

    for i in range(2*n+1):
        z_diff = Z_f[:,i] - zk_
        z_diff = np.reshape(z_diff,(6,1))
        Pzz = Pzz + weights[i]*np.matmul((z_diff),np.transpose(z_diff))

    Pzz = Pzz

    Pvv = Pzz + R_noise # innovation covariance

    zk_ = np.reshape(zk_,(6,1))

    return zk_, Pvv

def cross_cov(W_prime,Z,zk_,n,weights):
    Pxz = np.zeros((6,6))
    for i in range(2*n+1): 
        Zi = np.reshape(Z[:,i],(6,1))
        zk_ = np.reshape(zk_,(6,1))
        W_prime_i = np.reshape(W_prime[:,i],(6,1))
        Pxz = Pxz + weights[i]*np.matmul(W_prime_i,(Zi - zk_).T)

    Pxz = Pxz
    return Pxz


def process_IMU_data(IMU_data, imuparams):
    imu_raw_val = IMU_data["vals"]
    imu_ts = IMU_data["ts"][0]

    # gyro
    gyro_vals_raw = imu_raw_val[3:6, :]

    bg = np.average(gyro_vals_raw[:, :400], axis=1)
    bg = np.reshape(bg, (3, 1))

    # actual gyro values
    gyro_vals = (gyro_vals_raw - bg) * (0.3) * (pi / 180) * (3300 / 1023)

    # gyro is in wz wx wy format, convert to wx wy wz


    temp = np.copy(gyro_vals[0, :])
    gyro_vals[0, :] = gyro_vals[1, :]
    gyro_vals[1, :] = gyro_vals[2, :]
    gyro_vals[2, :] = temp

    # print("gyro_vals",gyro_vals[:,2000])


    g = 1
    acc_vals_raw = imu_raw_val[0:3, :]
    acc_vals = np.zeros((3, acc_vals_raw.shape[1]))
    acc_vals[0, :] = ( acc_vals_raw[0, :] * imuparams[0, 0] + imuparams[1, 0]) * g
    acc_vals[1, :] = ( acc_vals_raw[1, :] * imuparams[0, 1] + imuparams[1, 1]) * g
    acc_vals[2, :] = ( acc_vals_raw[2, :] * imuparams[0, 2] + imuparams[1, 2]) * g

    return gyro_vals, acc_vals, imu_ts

def check_positive_definite(A):
    is_symmetric = np.allclose(A, A.T)

    print("Is symmetric: ", is_symmetric)

    # Attempt to compute the Cholesky decomposition
    try:
        np.linalg.cholesky(A)
        is_positive_definite = True
    except np.linalg.LinAlgError:
        is_positive_definite = False

    # Check the results
    if is_symmetric and is_positive_definite:
        print("A is positive definite.")
    else:
        print("A is not positive definite.")


######################################################################################################################

# import Vicon and IMU data which are in .mat format

def load_data(file_num,is_test):
    imu_data_loc = '../Data/Train/IMU/imuRaw' + str(file_num) + '.mat'
    vicon_data_loc = '../Data/Train/Vicon/viconRot' + str(file_num) + '.mat'
    IMU_data = io.loadmat(imu_data_loc)
    if is_test==False:
        vicon_data = io.loadmat(vicon_data_loc)
    else:
        vicon_data=np.eye(3)
    imu_params = io.loadmat("../IMUParams.mat")

    # process IMU data
    gyro_vals, acc_vals, imu_ts = process_IMU_data(IMU_data, imu_params["IMUParams"])

    return gyro_vals, acc_vals, imu_ts, vicon_data

# IMU_data = io.loadmat('../Data/Train/IMU/imuRaw1.mat')
# vicon_data = io.loadmat('../Data/Train/Vicon/viconRot1.mat')
# imu_params = io.loadmat("../IMUParams.mat")

# # process IMU data
# gyro_vals, acc_vals, imu_ts = process_IMU_data(IMU_data, imu_params["IMUParams"])

######################################################################################################################

def ukf_main(file_num,is_test=False):

    gyro_vals, acc_vals, imu_ts, vicon_data = load_data(file_num,is_test)

    #Orientation store 
    ukf_euler_mat = np.zeros((3,len(imu_ts)))

    #initializing state vector
    if is_test==False:
        r = R.from_matrix(vicon_data['rots'][:,:,0])
    else:
        r=R.from_matrix(np.eye(3))
    q = r.as_quat()
    # quaternion needs to be in w,x,y,z format, it is now in x,y,z,w format
    q = [q[3],q[0],q[1],q[2]]
    print(q)
    x_hat = np.zeros((7,1),dtype=float)
    x_hat[0:4] = np.reshape(q,(4,1))

    #initializing covariance matrix
    Pkk = 1.0*np.identity(6) 

    #initializing process noise covariance matrix
    Q =  np.block([[ 55*(np.eye(3)), np.zeros((3, 3))],[np.zeros((3, 3)), 0.05*(np.eye(3))]])

    #initializing measurement noise covariance matrix
    R_noise = np.block([[ ( 20.0*np.eye(3)), np.zeros((3, 3))], [np.zeros((3, 3)), 2e-6 * (np.eye(3))]])

    #Number of sigma points
    n = 6

    #weight vector
    lamda = 3-n
    weights = np.zeros((2*n+1),dtype=float)
    weights[0] = lamda/(2*n+1)
    weights[1:] = 1/(2*n+1)
    

    for i in range(len(imu_ts)):

        print(i)


        if i == 0:
            dt = 0.01
        else:
            dt = imu_ts[i] - imu_ts[i-1]   


        #Generating sigma point parameters
        # check_positive_definite(Pkk)

        Xi = sigma_point(x_hat,Pkk,Q) 

        # Propagating the sigma points through the non linear process model to get the predicted sigma points
        # Y_pred = A(Xk,0) where A : w_k+1 = w_k
        Y_pred = propogate_sigma_points(Xi,n,dt)
        # print("Y_pred",Y_pred)

        # Calculating the mean of the predicted sigma points
        x_hat_, Pk_, W_prime = calc_mean_covariance(Y_pred,n,weights)
        # print("x_hat_",x_hat_)

        #Measurement prediction
        Z = measurement_prediction(Y_pred,n)

        #Calculate the mean and covariance of measurement sigma points
        zk_, Pvv = measurement_mean_cov(Z,R_noise,n,weights)

        # Calculating the cross covariance of the measurement sigma points and the predicted sigma points
        Pxz = cross_cov(W_prime,Z,zk_,n,weights)

        # Kalman gain
        Kk = np.matmul(Pxz,np.linalg.inv(Pvv)) 

        # measurement
        gyro = gyro_vals[:,i]
        acc = acc_vals[:,i]

        # measurement vector
        zk = np.concatenate((acc,gyro),axis=0)
        zk = np.reshape(zk,(6,1))


        # innovation
        vk = zk - zk_

        # state update
        innovation =  np.matmul(Kk,vk)
        innovation_quaternion = rot2quat(innovation[0:3])
        xhat_quat = quatmult(np.ravel(x_hat_[0:4]),np.ravel(innovation_quaternion))
        x_hat[0:4] = np.reshape(xhat_quat,(4,1))
        x_hat[4:7] = x_hat_[4:7,:] + innovation[3:6,:]

        # covariance update
        Pkk = Pk_ - np.matmul(np.matmul(Kk,Pvv),Kk.T)
        scipy_quat = [x_hat[1],x_hat[2],x_hat[3],x_hat[0]]
        orientation_quaternion = R.from_quat(np.ravel(scipy_quat))
        orientation_euler = orientation_quaternion.as_euler('xyz', degrees=False)
        # print(orientation_euler)
        ukf_euler_mat[:,i] = orientation_euler

    return ukf_euler_mat





#     ####################################################################################################################


#     vicon_r = []

#     vicon_time = []

#     for i in range(vicon_data['rots'].shape[2]):
#         r = R.from_matrix(vicon_data['rots'][:,:,i])
#         if np.linalg.norm(r.as_quat()) > 0:
#             vicon_time.append(vicon_data['ts'][0,i])
#             vicon_r.append(r)


#     vicon_r = R.concatenate(vicon_r)

#     valid_timestamp_range = [vicon_data['ts'][0,0], vicon_data['ts'][0,-1]]

#     # Ensure IMU timestamps are within the valid timestamp range
#     vicon_timestamps = np.clip(imu_ts, valid_timestamp_range[0], valid_timestamp_range[1])

#     slerp = Slerp(vicon_time, vicon_r)
#     vicon_r = slerp(vicon_timestamps)

#     #vicon_r is scipy.spatial.transform_rotation.Rotation object it doesn't have shape attribute


#     orientation_vicon = np.zeros((3, len(vicon_r)))

#     for i in range(len(vicon_r)):
#         orientation_vicon[:,i] = vicon_r[i].as_euler('ZYX', degrees=False)


#     # Plotting the orientation from acc and gyro and complementary filter and vicon

#     fig = plt.figure()
#     # three plots for each orientation
#     ax = fig.add_subplot(221)
#     ax.plot( ukf_euler_mat[0,:], label='$\phi_{comp}$')
#     ax.plot(orientation_vicon[0,:], label='$\phi_{vicon}$')
#     ax.set_title("$\phi$")
#     ax.legend(loc='upper right')

#     ax2 = fig.add_subplot(222)
#     ax2.plot( ukf_euler_mat[1,:], label='$\\theta_{comp}$')
#     ax2.plot( orientation_vicon[1,:], label='$\\theta_{vicon}$')
#     ax2.set_title("$\\theta$")
#     ax2.legend(loc='upper right')

#     ax3 = fig.add_subplot(223)
#     ax3.plot( ukf_euler_mat[2,:], label='$\psi_{comp}$')
#     ax3.plot( orientation_vicon[2,:], label='$\psi_{vicon}$')
#     ax3.set_title("$\psi$")
#     ax3.legend(loc='upper right')


#     plt.show()

#     #####################################################################################################################

# if __name__ == "__main__":
#     ukf_main(file_num=1)