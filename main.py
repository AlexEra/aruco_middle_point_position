import cv2
import pickle
from utils import detect_show_marker, undistort_image


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters_create()

    ''' Load coefficients '''
    with open('cam_param.pkl', 'rb') as f:
        camera_param = pickle.load(f)
    camera_mtx, dist_coefficients, _, _, _, _ = camera_param

    while True:
        ''' Capture frame-by-frame '''
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        ''' Undistorting '''
        # u_img = undistort_image(img, camera_mtx, dist_coefficients)
        
        ''' Show detected marker '''
        detect_show_marker(img, gray, aruco_dict, parameters, camera_mtx, dist_coefficients)
        
        ''' Press esc for close '''
        if cv2.waitKey(5) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
