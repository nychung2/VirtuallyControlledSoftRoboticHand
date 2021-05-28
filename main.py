from termcolor import colored
import image_processing as ip
import hand_landmarks as points
import time
import serial

calibration_open = ip.take_image(name='Please take an Image of an Open Hand', image_name='cal_open.jpg')
calibration_fist = ip.take_image(name='Please take an Image of a Fist', image_name='cal_closed.jpg')

try:
    # tic = time.perf_counter()
    process_openhand = ip.ImageObject(calibration_open)
    process_closedhand = ip.ImageObject(calibration_fist)
    angle1 = process_openhand.get_anglevector()
    angle2 = process_closedhand.get_anglevector()
    calibration_vector = ip.calculate_reference(angle1, angle2)
    # toc = time.perf_counter()
    # print(f"Calibration Images Processed in {toc - tic:0.5f} seconds.")
except Exception as e:
    print("Calibration Failed.")
    print(e)

input('Waiting to Start. Press Enter')
while True:
    path = ip.take_image(name='Please make any Gesture', image_name='gesture.jpg')
    
    # tic=time.perf_counter()
    image_captured = None
    try:
        image_captured = ip.ImageObject(path)
    except:
        print(colored('Hand Recognition Failed', 'red'))
        print(colored('Please Try Again', 'red'))
        input('Press Enter to Continue...')
        continue
    # toc = time.perf_counter()
    # print(f"Processed the Gesture Image in: {toc - tic:0.5f} seconds.")

    handedness = str(image_captured.get_handedness())
    handedness_result = handedness.find("Left")
    if handedness_result != -1:
        print(colored('Please Retry with your Right Hand', 'red'))
        input('Press Enter to Continue...')
        continue
    # image_captured.get_annotated_image()
    image_captured.plot_hand()
    gesture_angles = image_captured.get_anglevector()
    position_vector = ip.determine_position(gesture_angles, calibration_vector)
        
    try:
        send_to_hand = ip.send_to_arduino(position_vector)
        if send_to_hand == 0:
            print(colored('That Gesture Is not Allowed. Please Try Again.', 'red'))
            input('Press Enter to Continue...')
            continue
        else:
            print(colored("Successful Port!", 'green'))
            input('Press Enter to Continue...')
    except Exception as e:
        print(e)
        print(colored("Port to Hand Failed", 'red'))
        input('Press Enter to Continue...')
    