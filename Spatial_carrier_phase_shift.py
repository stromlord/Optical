from Holography_Simulation import *


def SCPS(I, cycle, k, alpha=0, beta=np.pi / 2):
    '''
    Chan, P.H; Bryanston-Cross, P.J; Parker, S.C (1995):
    Spatial phase stepping method of fringe-pattern analysis.
    In Optics and Lasers in Engineering 23 (5), pp. 343–354.
    DOI: 10.1016/0143-8166(95)90976-J.
    :param hologram: 全息图
    :param cycle:    载波周期
    :return:         重建波前
    '''

    [M, N] = np.shape(I)

    ref = reference_beam(M, N, r / cycle, k, alpha=alpha, beta=beta)
    hologram = interfence(I_filter, ref)  # 干涉

    hologram_L = np.zeros((M, N), dtype=complex)  # 向左移动一像素
    hologram_L[:, 0:N - 1] = hologram[:, 1:N]
    hologram_L[:, -1] = hologram[:, 0]

    hologram_R = np.zeros((M, N), dtype=complex)  # 向右移动一像素
    hologram_R[:, 0] = hologram[:, -1]
    hologram_R[:, 1:N] = hologram[:, 0:N - 1]

    theta = 2 * np.pi / cycle
    S_C = (hologram_R + hologram_L - 2 * np.cos(theta) * hologram) / (2 - 2 * np.cos(theta))  # 重建波前
    S_R = hologram - S_C + 1j * (hologram_R - hologram_L) / (2 * np.sin(theta))  # 带有正向相移的重建波前
    S_L = hologram - S_C - 1j * (hologram_R - hologram_L) / (2 * np.sin(theta))  # 带有负向相移的重建波前

    spectrum_display(np.fft.fftshift(np.fft.fft2(S_L)))

    plt.figure()
    plt.subplot(131), plt.imshow(np.abs(S_C), 'gray'), plt.axis('off'), plt.title('Center')
    plt.subplot(132), plt.imshow(np.abs(S_L), 'gray'), plt.axis('off'), plt.title('Left')
    plt.subplot(133), plt.imshow(np.abs(S_R), 'gray'), plt.axis('off'), plt.title('Right')
    plt.show()

    return S_C, S_R, S_L


def AFTA(I, k, aperture, alpha=np.pi / 4, beta=np.pi / 4):
    '''
    Dong, Zhichao; Chen, Zhenyue (2018):
    Advanced Fourier transform analysis method for phase retrieval from a single-shot spatial carrier fringe pattern.
    In Optics and Lasers in Engineering 107, pp. 149–160.
    DOI: 10.1016/j.optlaseng.2018.03.033.
    :param I:           全息图
    :param aperture:    数值孔径
    :return:            相位
    '''

    [M, N] = np.shape(I)

    ref = reference_beam(M, N, r / cycle, k, alpha=alpha, beta=beta)
    hologram = interfence(I_filter, ref)  # 干涉

    I1 = hologram
    I2 = spectrum_pixel_shifting(hologram, 1, 0)    # x+1
    I3 = spectrum_pixel_shifting(hologram, 0, 1)    # y+1
    I4 = spectrum_pixel_shifting(hologram, 1, 1)    # x+1, y+1

    c1 = diffraction_order_filter(I1, aperture)
    c2 = diffraction_order_filter(I2, aperture)
    c3 = diffraction_order_filter(I3, aperture)
    c4 = diffraction_order_filter(I4, aperture)

    phase_w = 0.5*(np.arctan(-np.real(c4-c3)/np.imag(c4-c3)) + np.arctan(-np.real(c2-c1)/np.imag(c2-c1)))
    plt.figure()
    plt.imshow(np.abs(phase_w), 'gray'), plt.axis('off'), plt.title('Phase')
    plt.show()

    return phase_w


if __name__ == "__main__":

    # 读取图像
    I0 = cv2.imread('Imgs/lenna.bmp', 0)
    I0 = I0 / 255  # 归一化
    I0 = I0 * np.exp(1j * np.pi * I0 / np.max(I0))  # 添加相位
    [M, N] = np.shape(I0)

    r = 0.532e-3  # 波长
    k = 2 * np.pi / r  # 波矢
    dx = 6.4e-3  # 像素间隔
    cycle = 3  # 载波周期

    I_filter = low_filter(I0, M/2/cycle)  # 数值孔径
    f_filter = np.fft.fftshift(np.fft.fft2(I_filter))

    # [S_C, S_R, S_L] = SCPS(I_filter, cycle, k)        # SCPS
    phase = AFTA(I_filter, k, M/2/cycle)                # AFTA



