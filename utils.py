import cv2
import numpy as np
import math


def detect_show_marker(img, gray, aruco_dict, parameters, camera_matrix, dist_coeffs):
    detected_1, detected_2 = False, False
    i, j = None, None
    distance_1, distance_2 = None, None
    font = cv2.FONT_HERSHEY_SIMPLEX
    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    img = cv2.aruco.drawDetectedMarkers(img, corners, ids)
    if ids is not None:
        i = 6  # Id of aruco - reference system.
        j = 5  # Id of target aruco.
        for k in range(0, len(ids)):
            rvec, tvec, marker_points = cv2.aruco.estimatePoseSingleMarkers(corners[k], 0.045, camera_matrix, dist_coeffs)
            if ids[k] == i:
                img = cv2.aruco.drawAxis(img, camera_matrix, dist_coeffs, rvec, tvec, 0.05)
                m_0_rvec = rvec
                m_0_tvec = tvec
                distance_1 = tvec[0][0][2]
                detected_1 = True
            elif ids[k] == j:
                m_1_rvec = rvec
                m_1_tvec = tvec
                distance_2 = tvec[0][0][2]
                detected_2 = True
            if detected_1 and detected_2:
                """ 
                frvec - orientation vector of the marker regarding the reference 
                system.
                ftvec -  position of aruco regarding the rs.
                m_1_tvec - position of aruco regarding camera.
                """
                frvec, ftvec = relation_position(m_1_rvec, m_1_tvec, m_0_rvec, m_0_tvec, False)
                for i in len(ftvec):
                    ftvec[0][i] /= 2  # ??

                frvec_new, ftvec_new = relation_position(frvec, ftvec, m_1_rvec, m_1_tvec, True)
                pose, orientation = inverse_vec(frvec_new, ftvec_new)

                cv2.putText(img, 'middle_point: %.2fsm' % (distance_1*100), (0, 32), font, 1, (0, 255, 0), 2,
                            cv2.LINE_AA)

    if distance_1 is not None:
        cv2.putText(img, 'Id' + str(i) + ' %.2fsm' % (distance_1*100), (0, 64), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    if distance_2 is not None:
        cv2.putText(img, 'Id' + str(j) + ' %.2fsm' % (distance_2*100), (0, 104), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    return cv2.imshow('frame', img)  # Final img.


def rotmtx_to_euler_angles(R):
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6 
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else :
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z])


def relation_position(rvec_1, tvec_1, rvec_2, tvec_2, is_inv):
    rvec_1, tvec_1 = rvec_1.reshape((3, 1)), tvec_1.reshape((3, 1))
    rvec_2, tvec_2 = rvec_2.reshape((3, 1)), tvec_2.reshape((3, 1))
    irvec, itvec = inverse_vec(rvec_2, tvec_2)
    if is_inv:
        mtx = cv2.composeRT(irvec, itvec, rvec_1, tvec_1)
    else:
        mtx = cv2.composeRT(rvec_1, tvec_1, irvec, itvec)
    composed_rvec, composed_tvec = mtx[0], mtx[1]
    composed_rvec = composed_rvec.reshape((3, 1))
    composed_tvec = composed_tvec.reshape((3, 1))
    return composed_rvec, composed_tvec


def inverse_vec(rvec, tvec):
    r, _ = cv2.Rodrigues(rvec)
    r = np.ndarray(r).T  # np.matrix(R).T
    i_tvec = np.dot(r, np.ndarray(-tvec))
    i_rvec, _ = cv2.Rodrigues(r)
    return i_rvec, i_tvec


def undistort_image(img, camera_matrix, dist_coeffs):
    h, w = img.shape[:2]
    new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))

    ''' Undistort '''
    dst = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_mtx)

    ''' Crop the image '''
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    return dst
