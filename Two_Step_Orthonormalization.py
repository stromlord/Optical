from Holography_Simulation import *


def Gram_Schmidt_Orthonormalization(I_filter, k, apecture, alpha=np.pi/9, beta=np.pi/2, shift_phase=np.pi/4):
    '''
    Vargas, J.; Quiroga, J. Antonio; Sorzano, C. O. S.; Estrada, J. C.; Carazo, J. M. (2012):
    Two-step demodulation based on the Gram-Schmidt orthonormalization method.
    In Optics letters 37 (3), pp. 443–445.
    DOI: 10.1364/OL.37.000443.
    :param I:   滤波后的全息图
    :return:    包裹的相位图
    '''
    ref1 = reference_beam(M, N, dx, k, alpha=alpha, beta=beta, shift_phase=0)
    ref2 = reference_beam(M, N, dx, k, alpha=alpha, beta=beta, shift_phase=shift_phase)

    hologram1 = interfence(I_filter, ref1)
    hologram2 = interfence(I_filter, ref2)

    hologram1_filter = high_filter(hologram1, apecture)
    hologram2_filter = high_filter(hologram2, apecture)

    # spectrum_display(np.fft.fftshift(np.fft.fft2(hologram1_filter)))

    u1 = hologram1_filter / inner_product(hologram1_filter, hologram1_filter)
    u2_p = hologram2_filter - inner_product(hologram2_filter, hologram1_filter) * u1
    u2 = u2_p / inner_product(hologram2_filter, hologram2_filter)

    phase = np.arctan(-u1 / u2)
    plt.figure()
    plt.subplot(121), plt.imshow(np.abs(phase), 'gray'), plt.axis('off'), plt.title('Wrapped Phase')
    plt.subplot(122), plt.plot(np.abs(phase[0, :])), plt.title('Plot')
    plt.show()


def inner_product(E1, E2):
    '''
    向量内积
    '''
    [M, N] = np.shape(E1)
    inner = 0
    for m in range(M):
        for n in range(N):
            inner = inner + E1[m, n] * E2[m, n]
    return inner


if __name__ == "__main__":
    # 读取图像
    I0 = cv2.imread('Imgs/lenna.bmp', 0)
    I0 = I0 / np.max(I0)  # 归一化
    I0 = np.exp(1j * np.pi * I0)  # 纯相位
    [M, N] = np.shape(I0)

    r = 0.532e-3
    k = 2 * np.pi / r
    dx = 6.4e-3

    I_filter = low_filter(I0, M / 8)  # 数值孔径

    Gram_Schmidt_Orthonormalization(I_filter, k, M / 8 + 40)

