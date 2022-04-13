import shutil
from pprint import pprint

import matplotlib.pyplot as plt
import cv2
import os


cv2.ocl.setUseOpenCL(False)

def drawLine(image, sx, sy):
    oy, ox = image.shape[:2]
    imRe = cv2.resize(image, (sx, sy))
    im = imRe.copy()
    ptList = []
    global mouseX, mouseY

    def setPoint(event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(im, (x, y), 5, (0, 0, 255), -1)
            mouseX, mouseY = x, y
            ptList.append((mouseX, mouseY))
            print(f'{mouseX}, {mouseY}')

    cv2.namedWindow('image')
    cv2.setMouseCallback('image', setPoint)
    sList = []
    while True:
        cv2.imshow('image', im)
        k = cv2.waitKey(20) & 0xFF
        if k == ord('x'):
            return sList
        # If 'r' key is pressed
        elif k == ord('r'):
            im = cv2.rotate(imRe, 1)
        # If 'c' key is pressed
        elif k == ord('c'):
            # Reset
            im = imRe.copy()
            ptList = []
            sList = []

        # If 'y' key is pressed and 4 points chosen
        elif k == ord('y') and len(ptList) == 4:
            # Extract rescaled list of corner points
            sList = [(int(pt[0] * (ox/sx)), int(pt[1] * (oy/sy))) for pt in ptList]
            draw_box(im, ptList)
            return sList

        # If 't' key is pressed and 2 points chosen
        elif k == ord('t') and len(ptList) == 2:
            # Extract rescaled list of corner points
            sList = [(int(pt[0] * (ox / sx)), int(pt[1] * (oy / sy))) for pt in ptList]
            return sList

        elif k == ord('u') and len(ptList) == 4:
            # Extract rescaled list of corner points
            sList = [(int(pt[0] * (ox / sx)), int(pt[1] * (oy / sy))) for pt in ptList]
            cv2.destroyAllWindows()
            draw_box(im, ptList)
            return sList

        elif k != 255 and (ord('0') <= k <= ord('9') or chr(k).isalpha()):
            # Extract rescaled list of corner points
            sList.append(chr(k))

        draw_box(im, ptList)


def draw_box(im, ptList):
    if len(ptList) == 4:
        cv2.line(im, ptList[0], ptList[1], (0, 255, 255), 2)
        cv2.line(im, ptList[1], ptList[2], (0, 255, 255), 2)
        cv2.line(im, ptList[2], ptList[3], (0, 255, 255), 2)
        cv2.line(im, ptList[3], ptList[0], (0, 255, 255), 2)


def main():
    # Assumes all notes in folder are the same denom
    labels = []
    dest = []
    for idx, note_dir in enumerate(os.listdir(notes_directory)):
        if idx == 0:
            denom_string, rgb_front_dir = get_series(note_dir, back=False)
            denom_string = denom_string + '_'
            print(denom_string)

        series_string, rgb_front_dir = get_series(note_dir, back=False)
        if series_string == '':
            series_string, rgb_front_dir = get_series(note_dir, back=True)
        print(series_string)

        dest.append(rgb_front_dir)
        labels.append(denom_string + series_string)

    pprint(labels)
    cont = input('Continue? (y/n)')
    if cont.lower() == 'n':
        return

    for note_dir, label in zip(dest, labels):
        shutil.move(note_dir, note_dir)

def get_series(note_dir, back):
    note, rgb_front_dir = get_note(note_dir, back=back)
    note = cv2.resize(note, (int(round(note.shape[1] / 10)), int(round(note.shape[0] / 10))))
    ptList = drawLine(note, note.shape[1], note.shape[0])
    series_string = ''.join(ptList)
    return series_string, rgb_front_dir


def get_note(note_dir, back=False):
    rgb_front_dir = notes_directory + note_dir + f'/{note_dir}_RGB_0.bmp'

    if back:
        rgb_front_dir = notes_directory + note_dir + f'/{note_dir}_RGB_1.bmp'

    note = cv2.imread(rgb_front_dir)
    if not note:
        rgb_front_dir = rgb_front_dir.replace('_0.bmp', '_front.bmp').replace('_1.bmp', '_back.bmp')
        note = cv2.imread(rgb_front_dir)
    return note, rgb_front_dir


if __name__ == '__main__':
    notes_directory = 'D:/DSC_OUTPUT/COMPLETE/' # Directory containing sub-directories /1/1_RGB_0.bmp
    main()