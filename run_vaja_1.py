import numpy as np
import cv2
import matplotlib.pyplot as plt
import ex1_utils as utils
from of_methods import lucaskanade, hornschunck, lucaskanade_improved, hornschunck_speedup

# Random noise image, rotated by -1 degree, Lucas-Kanade method

im1 = np.random.rand(200, 200).astype(np.float32)
im2 = im1.copy()
im2 = utils.rotate_image(im2, -1)

U_lk, V_lk = lucaskanade(im1, im2, 100)

fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))
ax1.imshow(im1)
ax2.imshow(im2)
utils.show_flow(U_lk, V_lk, ax3, type="angle")
utils.show_flow(U_lk, V_lk, ax4, type="field", set_aspect=True)
ax1.set_xticks([])
ax1.set_yticks([])
ax2.set_xticks([])
ax2.set_yticks([])
ax3.set_xticks([])
ax3.set_yticks([])
ax4.set_xticks([])
ax4.set_yticks([])
plt.show()

# Random noise image, rotated by -1 degree, Horn-Schunck method

im1 = np.random.rand(200, 200).astype(np.float32)
im2 = im1.copy()
im2 = utils.rotate_image(im2, -1)

U_hs, V_hs = hornschunck(im1, im2, 1000, 20)

fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))
ax1.imshow(im1)
ax2.imshow(im2)
utils.show_flow(U_hs, V_hs, ax3, type="angle")
utils.show_flow(U_hs, V_hs, ax4, type="field", set_aspect=True)
ax1.set_xticks([])
ax1.set_yticks([])
ax2.set_xticks([])
ax2.set_yticks([])
ax3.set_xticks([])
ax3.set_yticks([])
ax4.set_xticks([])
ax4.set_yticks([])
plt.show()

# Testing Lucas-Kanade method and Horn-Schunck method on "helly"

im1 = cv2.imread("other/helly1.jpeg", cv2.IMREAD_GRAYSCALE).astype(np.float32)
im1 = cv2.resize(im1, (500, 200))
im1 = utils.gausssmooth(im1, 1.5)
im2 = cv2.imread("other/helly2.jpeg", cv2.IMREAD_GRAYSCALE).astype(np.float32)
im2 = cv2.resize(im2, (500, 200))
im2 = utils.gausssmooth(im2, 1.5)

U_lk, V_lk = lucaskanade(im1, im2, 50)
U_hs, V_hs = hornschunck(im1, im2, 1000, 100)

fig1, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2,)
ax1.imshow(im1)
ax2.imshow(im2)
utils.show_flow(U_lk, V_lk, ax3, type="angle")
utils.show_flow(U_lk, V_lk, ax4, type="field", set_aspect=True)
utils.show_flow(U_hs, V_hs, ax5, type="angle")
utils.show_flow(U_hs, V_hs, ax6, type="field", set_aspect=True)
ax1.set_xticks([])
ax1.set_yticks([])
ax2.set_xticks([])
ax2.set_yticks([])
ax3.set_xticks([])
ax3.set_yticks([])
ax4.set_xticks([])
ax4.set_yticks([])
ax5.set_xticks([])
ax5.set_yticks([])
ax6.set_xticks([])
ax6.set_yticks([])
plt.show()

# Testing Lucas-Kanade method and Horn-Schunck method on "oppenheimer"

im1 = cv2.imread("other/oppie1.png", cv2.IMREAD_GRAYSCALE).astype(np.float32)
im1 = cv2.resize(im1, (500, 300))
im1 = utils.gausssmooth(im1, 1.5)
im2 = cv2.imread("other/oppie2.png", cv2.IMREAD_GRAYSCALE).astype(np.float32)
im2 = cv2.resize(im2, (500, 300))
im2 = utils.gausssmooth(im2, 1.5)

U_lk, V_lk = lucaskanade(im1, im2, 50)
U_hs, V_hs = hornschunck(im1, im2, 1000, 500)

fig1, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2,)
ax1.imshow(im1)
ax2.imshow(im2)
utils.show_flow(U_lk, V_lk, ax3, type="angle")
utils.show_flow(U_lk, V_lk, ax4, type="field", set_aspect=True)
utils.show_flow(U_hs, V_hs, ax5, type="angle")
utils.show_flow(U_hs, V_hs, ax6, type="field", set_aspect=True)
ax1.set_xticks([])
ax1.set_yticks([])
ax2.set_xticks([])
ax2.set_yticks([])
ax3.set_xticks([])
ax3.set_yticks([])
ax4.set_xticks([])
ax4.set_yticks([])
ax5.set_xticks([])
ax5.set_yticks([])
ax6.set_xticks([])
ax6.set_yticks([])
plt.show()

# Testing Lucas-Kanade method and Horn-Schunck method on "flow"

im1 = cv2.imread("other/flow5.png", cv2.IMREAD_GRAYSCALE).astype(np.float32)
im1 = cv2.resize(im1, (500, 300))
im1 = utils.gausssmooth(im1, 1.5)
im2 = cv2.imread("other/flow6.png", cv2.IMREAD_GRAYSCALE).astype(np.float32)
im2 = cv2.resize(im2, (500, 300))
im2 = utils.gausssmooth(im2, 1.5)

U_lk, V_lk = lucaskanade(im1, im2, 30)
U_hs, V_hs = hornschunck(im1, im2, 1000, 200)

fig1, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2,)
ax1.imshow(im1)
ax2.imshow(im2)
utils.show_flow(U_lk, V_lk, ax3, type="angle")
utils.show_flow(U_lk, V_lk, ax4, type="field", set_aspect=True)
utils.show_flow(U_hs, V_hs, ax5, type="angle")
utils.show_flow(U_hs, V_hs, ax6, type="field", set_aspect=True)
ax1.set_xticks([])
ax1.set_yticks([])
ax2.set_xticks([])
ax2.set_yticks([])
ax3.set_xticks([])
ax3.set_yticks([])
ax4.set_xticks([])
ax4.set_yticks([])
ax5.set_xticks([])
ax5.set_yticks([])
ax6.set_xticks([])
ax6.set_yticks([])
plt.show()

# Testing improved Lucas-Kanade method on "oppenheimer"

im1 = cv2.imread("other/oppie1.png", cv2.IMREAD_GRAYSCALE).astype(np.float32)
im1 = cv2.resize(im1, (500, 300))
im1 = utils.gausssmooth(im1, 1.5)
im2 = cv2.imread("other/oppie2.png", cv2.IMREAD_GRAYSCALE).astype(np.float32)
im2 = cv2.resize(im2, (500, 300))
im2 = utils.gausssmooth(im2, 1.5)

U_lk, V_lk = lucaskanade(im1, im2, 50)
U_hs, V_hs = lucaskanade_improved(im1, im2, 50)

fig1, ((ax5, ax6)) = plt.subplots(1, 2)
utils.show_flow(U_hs, V_hs, ax5, type="angle")
utils.show_flow(U_hs, V_hs, ax6, type="field", set_aspect=True)
ax5.set_xticks([])
ax5.set_yticks([])
ax6.set_xticks([])
ax6.set_yticks([])
plt.show()

# Testing Lucas-Kanade method on "oppenheimer" with different neighborhood sizes

im1 = cv2.imread("other/oppie1.png", cv2.IMREAD_GRAYSCALE).astype(np.float32)
im1 = cv2.resize(im1, (500, 300))
im1 = utils.gausssmooth(im1, 1.5)
im2 = cv2.imread("other/oppie2.png", cv2.IMREAD_GRAYSCALE).astype(np.float32)
im2 = cv2.resize(im2, (500, 300))
im2 = utils.gausssmooth(im2, 1.5)

U_lk, V_lk = lucaskanade(im1, im2, 25)
U_hs, V_hs = lucaskanade(im1, im2, 100)

fig1, ((ax3, ax4), (ax5, ax6)) = plt.subplots(2, 2,)
utils.show_flow(U_lk, V_lk, ax3, type="angle")
utils.show_flow(U_lk, V_lk, ax4, type="field", set_aspect=True)
utils.show_flow(U_hs, V_hs, ax5, type="angle")
utils.show_flow(U_hs, V_hs, ax6, type="field", set_aspect=True)
ax3.set_xticks([])
ax3.set_yticks([])
ax4.set_xticks([])
ax4.set_yticks([])
ax5.set_xticks([])
ax5.set_yticks([])
ax6.set_xticks([])
ax6.set_yticks([])
plt.show()

# Testing Horn-Schunck method on "oppenheimer" with different number of iterations

im1 = cv2.imread("other/oppie1.png", cv2.IMREAD_GRAYSCALE).astype(np.float32)
im1 = cv2.resize(im1, (500, 300))
im1 = utils.gausssmooth(im1, 1.5)
im2 = cv2.imread("other/oppie2.png", cv2.IMREAD_GRAYSCALE).astype(np.float32)
im2 = cv2.resize(im2, (500, 300))
im2 = utils.gausssmooth(im2, 1.5)

U_lk, V_lk = hornschunck(im1, im2, 100, 200)
U_hs, V_hs = hornschunck(im1, im2, 1000, 200)

fig1, ((ax3, ax4), (ax5, ax6)) = plt.subplots(2, 2,)
utils.show_flow(U_lk, V_lk, ax3, type="angle")
utils.show_flow(U_lk, V_lk, ax4, type="field", set_aspect=True)
utils.show_flow(U_hs, V_hs, ax5, type="angle")
utils.show_flow(U_hs, V_hs, ax6, type="field", set_aspect=True)
ax3.set_xticks([])
ax3.set_yticks([])
ax4.set_xticks([])
ax4.set_yticks([])
ax5.set_xticks([])
ax5.set_yticks([])
ax6.set_xticks([])
ax6.set_yticks([])
plt.show()

# Testing Horn-Schunck method on "oppenheimer" with different lambda values

im1 = cv2.imread("other/oppie1.png", cv2.IMREAD_GRAYSCALE).astype(np.float32)
im1 = cv2.resize(im1, (500, 300))
im1 = utils.gausssmooth(im1, 1.5)
im2 = cv2.imread("other/oppie2.png", cv2.IMREAD_GRAYSCALE).astype(np.float32)
im2 = cv2.resize(im2, (500, 300))
im2 = utils.gausssmooth(im2, 1.5)

U_lk, V_lk = hornschunck(im1, im2, 1000, 10)
U_hs, V_hs = hornschunck(im1, im2, 1000, 1000)

fig1, ((ax3, ax4), (ax5, ax6)) = plt.subplots(2, 2,)
utils.show_flow(U_lk, V_lk, ax3, type="angle")
utils.show_flow(U_lk, V_lk, ax4, type="field", set_aspect=True)
utils.show_flow(U_hs, V_hs, ax5, type="angle")
utils.show_flow(U_hs, V_hs, ax6, type="field", set_aspect=True)
ax3.set_xticks([])
ax3.set_yticks([])
ax4.set_xticks([])
ax4.set_yticks([])
ax5.set_xticks([])
ax5.set_yticks([])
ax6.set_xticks([])
ax6.set_yticks([])
plt.show()

# Measuring time of Lucas-Kanade and Horn-Schunck methods on "oppenheimer"

im1 = cv2.imread("other/oppie1.png", cv2.IMREAD_GRAYSCALE).astype(np.float32)
im1 = cv2.resize(im1, (500, 300))
im1 = utils.gausssmooth(im1, 1.5)
im2 = cv2.imread("other/oppie2.png", cv2.IMREAD_GRAYSCALE).astype(np.float32)
im2 = cv2.resize(im2, (500, 300))
im2 = utils.gausssmooth(im2, 1.5)

import time
start_time = time.time()
U_lk, V_lk = lucaskanade(im1, im2, 50)
end_time = time.time()
print("Lucas-Kanade method took {:.2f} seconds".format(end_time - start_time))

start_time = time.time()
U_hs, V_hs = hornschunck(im1, im2, 1000, 500)
end_time = time.time()
print("Horn-Schunck method took {:.2f} seconds".format(end_time - start_time))

# Testing sped up Horn-Schunck method on "oppenheimer"

im1 = cv2.imread("other/oppie1.png", cv2.IMREAD_GRAYSCALE).astype(np.float32)
im1 = cv2.resize(im1, (500, 300))
im1 = utils.gausssmooth(im1, 1.5)
im2 = cv2.imread("other/oppie2.png", cv2.IMREAD_GRAYSCALE).astype(np.float32)
im2 = cv2.resize(im2, (500, 300))
im2 = utils.gausssmooth(im2, 1.5)

U_lk, V_lk = lucaskanade(im1, im2, 25)

start_time = time.time()
U_hs, V_hs = hornschunck_speedup(im1, im2, 1000, 500, U_lk, V_lk)
end_time = time.time()
print("Improved Horn-Schunck method took {:.2f} seconds".format(end_time - start_time))

fig1, ((ax5, ax6)) = plt.subplots(1, 2,)
utils.show_flow(U_hs, V_hs, ax5, type="angle")
utils.show_flow(U_hs, V_hs, ax6, type="field", set_aspect=True)
ax5.set_xticks([])
ax5.set_yticks([])
ax6.set_xticks([])
ax6.set_yticks([])
plt.show()
