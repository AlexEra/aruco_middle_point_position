import cv2
import cv2.aruco as aruco
import numpy as np
import math
import transforms3d


def detect_show_marker(img, gray, aruco_dict, parameters, cameraMatrix,
                       distCoeffs):
    detected_1, detected_2 = False, False
    i, j = None, None
    distance_1, distance_2 = None, None
    font = cv2.FONT_HERSHEY_SIMPLEX
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict,
                                                          parameters=parameters)
    img = aruco.drawDetectedMarkers(img, corners, ids)
    if ids is not None:
        i = 6  # Id of aruco - reference system.
        j = 5  # Id of target aruco.
        for k in range(0, len(ids)):
            rvec, tvec, markerPoints = aruco.estimatePoseSingleMarkers(corners[k],
                                                                       0.045,
                                                                       cameraMatrix,
                                                                       distCoeffs)
            if ids[k] == i:
                img = aruco.drawAxis(img, cameraMatrix, distCoeffs, rvec, tvec, 0.05)
                m_0_rvec = rvec
                m_0_tvec = tvec
                distance_1 = tvec[0][0][2]
                detected_1 = True
            elif ids[k] == j:
                m_1_rvec = rvec
                m_1_tvec = tvec
                distance_2 = tvec[0][0][2]
                detected_2 = True
            if detected_1 && detected_2:
                frvec, ftvec = relation_position(m_1_rvec, m_1_tvec, m_0_rvec, m_0_tvec, False)
                """ 
                frvec - orientation vector of the marker regarding the reference 
                system.
                ftvec -  position of aruco regarding the rs.
                m_1_tvec - position of aruco regarding camera.
                """
                for i in len(ftvec):
                    ftvec[0][i] /= 2  # ??

                frvec_new, ftvec_new = relation_position(frvec, ftvec, m_1_rvec, m_1_tvec, True)
                frvec_new, ftvec_new = inverse_vec(frvec_new, ftvec_new)

                cv2.putText(img, 'middle_point: %.2fsm' % (distance_1*100), (0, 32), font, 
                    1, (0, 255, 0), 2, cv2.LINE_AA)            

    if (distance_1 is not None):
        cv2.putText(img, 'Id' + str(i) + ' %.2fsm' % (distance_1*100), (0, 64), font, 
                    1, (0, 255, 0), 2, cv2.LINE_AA)
    if (distance_2 is not None):
        cv2.putText(img, 'Id' + str(j) + ' %.2fsm' % (distance_2*100), (0, 104), font, 
                    1, (0, 255, 0), 2, cv2.LINE_AA)
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
    composedRvec, composedTvec = mtx[0], mtx[1]
    composedRvec = composedRvec.reshape((3, 1))
    composedTvec = composedTvec.reshape((3, 1))
    return composedRvec, composedTvec


def inverse_vec(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    R = np.matrix(R).T
    itvec = np.dot(R, np.matrix(-tvec))
    irvec, _ = cv2.Rodrigues(R)
    return irvec, itvec


def undistort_image(img, cameraMatrix, distCoeffs):
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, 
                                                      (w, h), 1,(w, h))
    # Undistort.
    dst = cv2.undistort(img, cameraMatrix, distCoeffs, None, newcameramtx)
    
    # Crop the image.
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    return dst
