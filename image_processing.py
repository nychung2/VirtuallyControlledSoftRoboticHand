import cv2
import time
import mediapipe as mp
from PIL import Image, ImageGrab
from termcolor import colored
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import serial
import hand_landmarks as points

''' This contains all the functions for the Soft-Robotics Hand. The main libraries used here are cv2, mediapipe, numpy, and matplotlib.'''

# Takes webcam image and stores it in the examples folder
def take_image(name, image_name):
    cv2.namedWindow(name)
    vc = cv2.VideoCapture(0)

    if vc.isOpened():
        rval, frame = vc.read()
    else:
        rval = False

    while rval:
        cv2.imshow(name, frame)
        rval, frame = vc.read()
        k = cv2.waitKey(20)
        if k%256 == 27:
            break
        elif k%256 == 32:
            cv2.imwrite('examples/' + image_name, frame)
            print(colored('Image Captured', 'green'))
            break

    vc.release()
    cv2.destroyAllWindows()
    return 'examples/' + image_name

# Takes screenshot of the computer screen
def take_screenshot():
    img = ImageGrab.grab(bbox=(0,0,1920,1080)) #x1, y1, x2, y2
    img_np = np.array(img)
    frame = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    image = 'screenshot_input_image.jpg'
    cv2.imwrite('examples/' + image, frame)
    while True:
        cv2.imshow("frame", frame)
        k = cv2.waitKey(20)
        if k%256 == 27:
            break
        elif k%256 == 32:
            break
    
    cv2.destroyAllWindows()
    return 'examples/' + image

# Calculates the Reference/Calibration Threshold Vector
def calculate_reference(open_hand, fist_hand):
    vector = np.add(open_hand,fist_hand)*(1.0/2.0)
    return vector

# Calulates Position of Hand. It is assumed all digits are open (1). It will then go through each digit and determine
# if the digit is curled (0) by comparing to the reference angle.
def determine_position(position, vector_reference_angle):
    print(colored('Calculating State of Digits', 'green'))
    return_vector = [1, 1, 1, 1, 1]
    print('Reference: ' + np.array_str(vector_reference_angle))
    print('Gesture: ' + np.array_str(position))
    for i in range(5):
        if position[i] < vector_reference_angle[i]:
            return_vector[i] = 0 # Engage Digit
    print('Hand Position: ' + str(return_vector))
    return return_vector 

# Sends data to Arduinos. Change the COM ports here.
def send_to_arduino(vector):
    print('Writing to hand...')
    # tic = time.perf_counter()
    var_digits = serial.Serial('COM4', 115200) # Soft Robotics Control Board
    var_thumb = serial.Serial('COM7', 115200) # Programmable Air Control Board
    time.sleep(2)
    print(str(vector))
    if str(vector) in points.banned_combinations:
        return 0
    thumb = vector.pop(0)
    count = np.count_nonzero(vector)
    if count is None:
        count = 0
    else:
        count = .7*(4 - count)
    if thumb == 0:
        var_thumb.write(bytes('l', 'utf-8'))
        time.sleep(.5)
        to_fingers = points.hand_combinations.get(str(vector))
        var_digits.write(bytes(to_fingers, 'utf-8'))
        time.sleep(count)
        var_digits.write(bytes('p', 'utf-8'))
        toc = time.perf_counter()
        print(f"Hand Response Time was {toc - tic:0.5f} seconds")
        return 1
    else: 
        to_fingers = points.hand_combinations.get(str(vector))
        var_digits.write(bytes(to_fingers, 'utf-8'))
        time.sleep(count)
        var_digits.write(bytes('p', 'utf-8'))
        # toc = time.perf_counter()
        # print(f"Hand Response Time was {toc - tic:0.5f} seconds")
        return 1

# Image Object that stores one hand as an object instance
class ImageObject:
    def __init__(self, image):
        self.image = image
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.landmarks = points.hand_list
        self.image_resized = self.resize_image(self.image, 100)
        self.hand = self.process_hand(self.image_resized)
        self.image_annotated = self.hand[0]
        self.hand_coordinates = self.hand[1]
        self.handedness
        self.hand_vectors = self.reformat_coordinate_data()
        self.angle_vector = self.determine_angles()

    # Scales input image
    def resize_image(self, image_input, scale_percent):
        image = cv2.imread(image_input)
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dsize = (width, height)
        output = cv2.resize(image, dsize)
        new_path = image_input.replace('.jpg', '_' + str(scale_percent) + '.jpg')
        cv2.imwrite(new_path, output)
        return new_path

    # MediaPipe Processing of Image
    def process_hand(self, image_input):
        print(colored('Processing Image', 'blue'))
        hands = self.mp_hands.Hands(
            static_image_mode = True,
            max_num_hands = 1,
            min_detection_confidence = .75, 
            min_tracking_confidence = .75
        )

        image = cv2.flip(cv2.imread(image_input), 1)
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        self.handedness = results.multi_handedness

        if not results.multi_hand_landmarks:
            return None
        image_height, image_width, _ = image.shape
        annotated_image = image.copy()
        coordinate_list = []
        for hand_landmarks in results.multi_hand_landmarks:
            for point in self.landmarks:
                x = hand_landmarks.landmark[getattr(self.mp_hands.HandLandmark, point)].x
                y = hand_landmarks.landmark[getattr(self.mp_hands.HandLandmark, point)].y
                z = hand_landmarks.landmark[getattr(self.mp_hands.HandLandmark, point)].z
                coordinates = [x, y, z]
                coordinate_list.append(coordinates)
            self.mp_drawing.draw_landmarks(annotated_image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        hands.close()
        annotated_image = cv2.flip(annotated_image, 1)
        path = self.image.replace('.jpg', '_annotated.jpg')
        cv2.imwrite(path, annotated_image)
        return path, coordinate_list

    # Restructures Hand coordinate Data
    def reformat_coordinate_data(self):
        print(colored('Reformatting Coordinate Data Structure'))
        thumb_x = [self.hand_coordinates[0][0]]
        thumb_y = [self.hand_coordinates[0][1]]
        thumb_z = [self.hand_coordinates[0][2]]
        for i in range(1,5):
            thumb_x.append(self.hand_coordinates[i][0])
            thumb_y.append(self.hand_coordinates[i][1])
            thumb_z.append(self.hand_coordinates[i][2])
        thumb = [thumb_x, thumb_y, thumb_z]
            
        index_x = [self.hand_coordinates[0][0]]
        index_y = [self.hand_coordinates[0][1]]
        index_z = [self.hand_coordinates[0][2]]
        for i in range(5,9):
            index_x.append(self.hand_coordinates[i][0])
            index_y.append(self.hand_coordinates[i][1])
            index_z.append(self.hand_coordinates[i][2])
        index = [index_x, index_y, index_z]
        
        middle_x = [self.hand_coordinates[0][0]]
        middle_y = [self.hand_coordinates[0][1]]
        middle_z = [self.hand_coordinates[0][2]]
        for i in range(9,13):
            middle_x.append(self.hand_coordinates[i][0])
            middle_y.append(self.hand_coordinates[i][1])
            middle_z.append(self.hand_coordinates[i][2])
        middle = [middle_x, middle_y, middle_z]

        ring_x = [self.hand_coordinates[0][0]]
        ring_y = [self.hand_coordinates[0][1]]
        ring_z = [self.hand_coordinates[0][2]]
        for i in range(13,17):
            ring_x.append(self.hand_coordinates[i][0])
            ring_y.append(self.hand_coordinates[i][1])
            ring_z.append(self.hand_coordinates[i][2])
        ring = [ring_x, ring_y, ring_z]
        
        pinky_x = [self.hand_coordinates[0][0]]
        pinky_y = [self.hand_coordinates[0][1]]
        pinky_z = [self.hand_coordinates[0][2]]
        for i in range(17,21):
            pinky_x.append(self.hand_coordinates[i][0])
            pinky_y.append(self.hand_coordinates[i][1])
            pinky_z.append(self.hand_coordinates[i][2])
        pinky = [pinky_x, pinky_y, pinky_z]

        return_vector = [thumb, index, middle, ring, pinky]
        return return_vector

    # Returns Vector with x,y,z coordinates of digit
    def get_digit(self, i):
        digit_dict = {
            '0': 'Thumb',
            '1': 'Index Finger',
            '2': 'Middle Finger',
            '3': 'Ring Finger',
            '4': 'Pinky Finger',
        }
        return digit_dict.get(str(i))    
    
    # Shows 3D Plot of Hand
    def plot_hand(self):
        print(colored('Plotting Hand', 'blue'))
        hand_vectors = self.hand_vectors
        fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
        for i in range(5):
            plt.plot(hand_vectors[i][0], hand_vectors[i][1], hand_vectors[i][2], 'o-', label = self.get_digit(i))
        plt.legend()
        plt.show()

    def determine_angles(self):
        i = 0
        angle_vector = np.array([0, 0, 0, 0, 0])
        for digit in self.hand_vectors:
            if i == 0:
                v1 = [digit[0][1] - digit[0][2], digit[1][1] - digit[1][2], digit[2][1] - digit[2][2]]
                v2 = [digit[0][3] - digit[0][2], digit[1][3] - digit[1][2], digit[2][3] - digit[2][2]]
                unit_v1 = v1/np.linalg.norm(v1)
                unit_v2 = v2/np.linalg.norm(v2)
                dp = np.dot(unit_v1, unit_v2)
                angle_vector[i] = np.degrees(np.arccos(dp))
                i += 1
                continue

            v1 = [digit[0][0] - digit[0][1], digit[1][0] - digit[1][1], digit[2][0] - digit[2][1]]
            v2 = [digit[0][2] - digit[0][1], digit[1][2] - digit[1][1], digit[2][2] - digit[2][1]]
            unit_v1 = v1/np.linalg.norm(v1)
            unit_v2 = v2/np.linalg.norm(v2)
            dp = np.dot(unit_v1, unit_v2)
            angle_vector[i] = np.degrees(np.arccos(dp))
            i += 1
        return angle_vector

    # Shows input image
    def get_image(self):
        image = cv2.imread(self.image_resized)
        cv2.imshow(str(self.image_resized), image)
        cv2.waitKey(0)
        return None

    # Shows annotated image
    def get_annotated_image(self):
        image = cv2.imread(self.image_annotated)
        cv2.imshow(str(self.image_resized), image)
        cv2.waitKey(0)
        return None

    # Returns hand coordinates
    def get_hand_landmarks(self):
        return self.hand_coordinates

    # Returns handedness recognition
    def get_handedness(self):
        return self.handedness

    # Returns vector of hand
    def get_hand_vectors(self):
        return self.hand_vectors

    # Returns angles of hand
    def get_anglevector(self):
        return self.angle_vector
