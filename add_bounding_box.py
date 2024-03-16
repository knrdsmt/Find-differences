import cv2
import numpy as np

obraz1 = cv2.imread('dublin.jpg')
obraz2 = cv2.imread('dublin_edited.jpg')

roznica = np.abs(obraz1.astype(int) - obraz2.astype(int)).astype(np.uint8)
roznica_szarosc = np.dot(roznica[..., :3], [0.2989, 0.5870, 0.1140])
prog = (roznica_szarosc > 30) * 255

kontury = []
for y in range(prog.shape[0]):
    for x in range(prog.shape[1]):
        if prog[y, x] == 255:
            kontury.append((x, y))

obiekty = []
for kontur in kontury:
    dodano = False
    for obiekt in obiekty:
        if min([np.sqrt((x - kontur[0]) ** 2 + (y - kontur[1]) ** 2) for x, y in obiekt]) < 50:
            obiekt.append(kontur)
            dodano = True
            break
    if not dodano:
        obiekty.append([kontur])

for obiekt in obiekty:
    x_min = min(obiekt, key=lambda x: x[0])[0]
    y_min = min(obiekt, key=lambda x: x[1])[1]
    x_max = max(obiekt, key=lambda x: x[0])[0]
    y_max = max(obiekt, key=lambda x: x[1])[1]

    obraz2[y_min:y_max, x_min] = [0, 255, 0]
    obraz2[y_min:y_max, x_max] = [0, 255, 0]
    obraz2[y_min, x_min:x_max] = [0, 255, 0]
    obraz2[y_max, x_min:x_max] = [0, 255, 0]

cv2.imwrite('dublin_bbox.jpg', obraz2)

najwiekszy_obiekt = max(obiekty, key=len)

# Wyznacz bounding box dla największego obiektu
x_min = min(najwiekszy_obiekt, key=lambda x: x[0])[0] + 1
y_min = min(najwiekszy_obiekt, key=lambda x: x[1])[1] + 1
x_max = max(najwiekszy_obiekt, key=lambda x: x[0])[0]
y_max = max(najwiekszy_obiekt, key=lambda x: x[1])[1]

wyciecie = obraz2[y_min:y_max, x_min:x_max]
obraz3 = np.copy(wyciecie)
wyciecie1 = obraz1[y_min:y_max, x_min:x_max]
wyciecie2 = obraz2[y_min:y_max, x_min:x_max]

roznica_wyciecia = np.abs(wyciecie1.astype(int) - wyciecie2.astype(int)).astype(np.uint8)
roznica_wyciecia_szarosc = np.dot(roznica_wyciecia[..., :3], [0.2989, 0.5870, 0.1140])
prog_wyciecia = (roznica_wyciecia_szarosc > 7) * 255

roznice = []
for y in range(prog_wyciecia.shape[0]):
    for x in range(prog_wyciecia.shape[1]):
        if prog_wyciecia[y, x] == 255:
            roznice.append((x, y))

maska = np.zeros_like(obraz3[:, :, 0], dtype=np.uint8)
for roznica in roznice:
    maska[roznica[1], roznica[0]] = 1

obraz3_rgba = np.dstack((obraz3, np.full(obraz3.shape[:2], 255)))
obraz3_rgba[maska != 1] = [0, 0, 0, 0]
cv2.imwrite('bbox_bez_tla.png', obraz3_rgba)
