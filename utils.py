import cv2
import numpy as np


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
                img = cv2.aruco.drawAxis(img, camera_matrix, dist_coeffs, rvec, tvec, 0.05)
                m_1_rvec = rvec
                m_1_tvec = tvec
                distance_2 = tvec[0][0][2]
                detected_2 = True
            if detected_1 and detected_2:
                middle_point = np.array([(m_0_tvec[0][0][0] + m_1_tvec[0][0][0]) / 2,
                                         (m_0_tvec[0][0][1] + m_1_tvec[0][0][1]) / 2,
                                         (m_0_tvec[0][0][2] + m_1_tvec[0][0][2]) / 2])
                middle_point = middle_point.reshape((1, 1, 3))
                img = cv2.aruco.drawAxis(img, camera_matrix, dist_coeffs, rvec, middle_point, 0.05)
                cv2.putText(img, 'distance_to_platform: %.4fm' % (middle_point[0][0][2]), (0, 32), font, 1,
                            (0, 255, 0), 2, cv2.LINE_AA)

    if distance_1 is not None:
        cv2.putText(img, 'Id' + str(i) + ' %.4fm' % distance_1, (0, 64), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    if distance_2 is not None:
        cv2.putText(img, 'Id' + str(j) + ' %.4fm' % distance_2, (0, 104), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    if cv2.waitKey(5) == 115:
        print('saved')
        cv2.imwrite('test.png', img)

    return cv2.imshow('frame', img)  # Final img.


def undistort_image(img, camera_matrix, dist_coeffs):
    h, w = img.shape[:2]
    new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))

    ''' Undistort '''
    dst = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_mtx)

    ''' Crop the image '''
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    return dst
