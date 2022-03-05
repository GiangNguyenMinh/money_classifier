import cv2 as cv
import numpy as np
import os
import sys
import argparse

# ['0000', '1000', '5000', '20000', '50000']

def Gamma(img):

    if np.random.randint(2):
        gamma = np.random.uniform(0.7, 1.2)
        img = pow(img, float(gamma))
    return img
def main_run():
    lable = args.value
    link = './data/'
    if not os.path.exists(link):
        os.mkdir(link)
    if not os.path.exists(os.path.join(link, lable)):
        os.mkdir(os.path.join(link, lable))

    cap = cv.VideoCapture(0)
    i = 1
    while 1:
        i += 1
        ret, frame = cap.read()
        if not ret:
            break
        cv.imshow('window', frame)

        if i > 60 and i <= 1060:
            print('cap count:', i-60)
            frame = cv.resize(frame, dsize=(224, 224))
            frame = Gamma(frame)
            cv.imwrite(os.path.join(link, lable, '{}.png'.format(i-60)), frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='make data for money classifier')
    parser.add_argument('--value', type=str, default=0000, help='modify in [0000, 1000, 5000, 20000, 50000]')
    args = parser.parse_args()
    main_run()
