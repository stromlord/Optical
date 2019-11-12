from Holography_Simulation import *
from Diffraction import *
import cv2
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    I0 = cv2.imread("Imgs/lenna.bmp", 0)
    [M, N] = np.shape(I0)

    r = 0.532e-3  # 波长
    k = 2 * np.pi / r  # 波矢
    pixel = 6.4e-3  # 像素间隔

    z = 180
    I, _ = angular_diff(I0, z, pixel, r)

    ref = reference_beam(M, N, pixel, k)
    hologram = interfence(I, ref)

    hologram_fft = np.fft.fftshift(np.fft.fft2(hologram))

    plt.figure()
    plt.subplot(221), plt.imshow(np.abs(I0), 'gray')
    plt.subplot(222), plt.imshow(np.abs(I), 'gray')
    plt.subplot(223), plt.imshow(np.abs(hologram), 'gray')
    plt.subplot(224), plt.imshow(np.log(np.abs(hologram_fft)), 'gray')

    # _, I_res = angular_diff(hologram, -z, pixel, r)
    _, I_res, _ = fresnel_diff_s_fft(hologram, z, pixel, r)

    plt.show()

    cv2.namedWindow('Reconstruction')
    cv2.imshow('Reconstruction', 10*np.abs(I_res)/np.max(np.abs(I_res)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

