import cv2
import matplotlib.pyplot as plt
import numpy as np


def reference_beam(M, N, dx, k, alpha=np.pi/5, beta=np.pi/5, shift_phase=0):
    '''
    生成参考光
    :param M, N:        图像尺寸
    :param dx:          采样间隔
    :param k:           波矢
    :param alpha:       与x轴夹角
    :param beta:        与y轴夹角
    :param shift_phase: 相移值
    :return:            参考光
    '''
    x = np.linspace(-(dx * N) / 2, (dx * N) / 2, N)
    y = np.linspace(-(dx * M) / 2, (dx * M) / 2, M)
    [x, y] = np.meshgrid(x, y)

    ref = np.exp(1j * k * (x * np.cos(alpha) + y * np.cos(beta)))   # 参考光
    return ref*np.exp(1j * shift_phase)


def low_filter(img, r, dx=0, dy=0):
    '''
    理想圆形低通滤波
    :param img: 输入图像
    :param r:   滤波半径
    :param dx:  x方向距离
    :param dy:  y方向距离
    :return:    滤波后图像
    '''
    [M, N] = np.shape(img)
    f = np.fft.fftshift(np.fft.fft2(img))
    mask = np.zeros((M, N))
    for n in range(N):
        for m in range(M):
            if (n-N/2-dy)**2 + (m-M/2-dx)**2 <= r**2:
                mask[n, m] = 1
    f = f * mask
    img_filter = np.fft.ifft2(np.fft.ifftshift(f))
    return img_filter


def high_filter(img, r, dx=0, dy=0):
    '''
    理想圆形高通滤波器
    :param img:  输入图像
    :param r:    滤波半径
    :param dx:   x方向距离
    :param dy:   y方向距离
    :return:     滤波后图像
    '''
    [M, N] = np.shape(img)
    f = np.fft.fftshift(np.fft.fft2(img))
    mask = np.ones((M, N))
    for n in range(N):
        for m in range(M):
            if (n-N/2-dy)**2 + (m-M/2-dx)**2 <= r**2:
                mask[n, m] = 0
    f = f * mask
    img_filter = np.fft.ifft2(np.fft.ifftshift(f))
    return img_filter


def spectrum_pixel_shifting(spectrum, dx, dy):
    '''
    频谱像素移动
    :param spectrum: 频谱
    :param dx, dy:   移动距离
    :return:         移动后的频谱
    '''
    [M, N] = np.shape(spectrum)

    # x方向
    spectrum_shifting_x = np.zeros((M, N), dtype=complex)
    if dx > 0:              # 右移
        spectrum_shifting_x[:, 0:dx] = spectrum[:, N - dx:N]
        spectrum_shifting_x[:, dx:N] = spectrum[:, 0:N - dx]
    elif dx < 0:            # 左移
        dx = np.abs(dx)
        spectrum_shifting_x[:, 0:N - dx] = spectrum[:, dx:N]
        spectrum_shifting_x[:, N - dx:N] = spectrum[:, 0:dx]
    else:
        spectrum_shifting_x = spectrum

    # y方向
    spectrum_shifting_y = np.zeros((M, N), dtype=complex)
    if dy > 0:              # 下移
        spectrum_shifting_y[0:dy, :] = spectrum_shifting_x[M - dy:M, :]
        spectrum_shifting_y[dy:M, :] = spectrum_shifting_x[0:M - dy, :]
    elif dy < 0:            # 上移
        dy = np.abs(dy)
        spectrum_shifting_y[0:M - dy, :] = spectrum_shifting_x[dy:M, :]
        spectrum_shifting_y[M - dy:M, :] = spectrum_shifting_x[0:dy, :]
    else:
        spectrum_shifting_y = spectrum_shifting_x

    return spectrum_shifting_y


def interfence(I, R):
    '''
    干涉
    :param I: 物光
    :param R: 参考光
    :return:  全息图
    '''
    inter = I / np.max(I) + R
    II = inter * np.conj(inter)
    return II


def spectrum_display(spectrum, gray_degree=100, max=0,  title="Spectrum"):
    '''
    频谱显示
    :param spectrum:        频谱
    :param gray_degree:     灰度等级
    :param max:             归一化程度
    :param title:           标题
    '''
    if max == 0:
        display = gray_degree * np.abs(spectrum)/np.max(np.abs(spectrum))
    else:
        display = gray_degree * np.abs(spectrum)/max
    cv2.imshow(title, display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def order_location(img, degree_x=50, degree_y=50, mode='Left'):
    '''衍射级次中心位置'''
    [M, N] = np.shape(img)
    f_II = np.fft.fftshift(np.fft.fft2(img))

    if mode == 'Left':
        [xm, ym] = np.where(abs(f_II) == np.max(abs(f_II[0:M // 2 - degree_y, 0:N // 2 - degree_x])))
    elif mode == 'Right':
        [xm, ym] = np.where(abs(f_II) == np.max(abs(f_II[0:M // 2 - degree_y, N // 2 + degree_x:N])))

    dx = xm[0] - M // 2
    dy = ym[0] - N // 2
    return xm[0], ym[0], dx, dy


def diffraction_order_filter(hologram, aperture, degree_x=50, degree_y=50):
    '''
    滤出衍射级次
    :param hologram: 全息图
    :param aperture: 数值孔径
    :return:         重建波前
    '''
    [xm, ym, dx, dy] = order_location(hologram, degree_x, degree_y)
    holo_filter = low_filter(hologram, aperture, dx, dy)
    shifting_spectrum = spectrum_pixel_shifting(np.fft.fftshift(np.fft.fft2(holo_filter)), -dx, -dy)
    wavefront = np.fft.ifft2(np.fft.ifftshift(shifting_spectrum))
    return wavefront


def four_step_phase_shift(I0, I1, I2, I3, R, theta1=np.pi/2, theta2=np.pi, theta3=3*np.pi/2):
    '''
    四步相移
    :param I0, I1, I3, I4:           输入4幅全息图
    :param R:                        参考光
    :param theta1， theta2, theta3:  相移值
    :return:                         重建波前
    '''
    if theta1 == np.pi/2 and theta2 == np.pi and theta3 == 3*np.pi/2:
        U = (I0 - I2 + 1j * (I1 - I3)) / (4 * np.conj(R))
    else:
        U_up = (I0-I2)/(1-np.exp(1j*theta2))-(I1-I3)/(np.exp(1j*theta1)-np.exp(1j*theta3))
        U_down = np.conj(R)*(1/(np.exp(1j*(theta1+theta3)))-np.exp(1j*theta2))
        U = U_up/U_down
    return U


def three_step_phase_shift(I0, I1, I2, R, theta1=np.pi/2, theta2=np.pi):
    '''
    三步相移
    :param I0, I1, I2:     输入全息图
    :param R:              参考光
    :param theta1, theta2: 相移值
    :return:               重建波前
    '''
    if theta1 == np.pi/2 and theta2 == np.pi:
        U = (1+1j)*(I1-I2+1j*(I1-I0))/(4*np.conj(R))
    else:
        U_up = (I0-I2)/(1-np.exp(1j*theta2))-(I0-I1)/(1-np.exp(1j*theta1))
        U_down = np.conj(R)*(1/np.exp(1j*theta1)-1/np.exp(1j*theta2))
        U = U_up/U_down
    return U


if __name__ == "__main__":
    I0 = cv2.imread('Imgs/lenna.bmp', 0)
    I0 = I0/255                             # 归一化
    I0 = I0*np.exp(1j*2*np.pi*I0/np.max(I0))  # 添加相位
    [M, N] = np.shape(I0)

    r = 0.532e-3            # 波长
    k = 2 * np.pi / r       # 波矢
    pixel = 6.4e-3          # 像素间隔
    aperture = 60           # 数值孔径

    I_filter = low_filter(I0, aperture)  # 数值孔径
    f_filter = np.fft.fftshift(np.fft.fft2(I_filter))

    '''normal hologram'''
    ref = reference_beam(M, N, pixel, k)                    # 生成参考光
    hologram = interfence(I_filter, ref)                    # 物光，参考光干涉
    hologram_fft = np.fft.fftshift(np.fft.fft2(hologram))   # 频谱

    # spectrum_display(hologram_fft)

    plt.figure()
    plt.imshow(np.abs(hologram), 'gray'), plt.axis('off'), plt.title('Hologram')
    # plt.show()

    '''FFT'''
    wavefront = diffraction_order_filter(hologram, aperture)
    holo_amplitude = np.abs(wavefront)
    holo_phase = np.angle(wavefront)
    # spectrum_display(holo_shifting, max=np.max(np.abs(hologram_fft)))
    plt.figure()
    plt.subplot(121), plt.imshow(holo_amplitude, 'gray'), plt.axis('off'), plt.title('Amplitude')
    plt.subplot(122), plt.imshow(holo_phase, 'gray'), plt.axis('off'), plt.title('Phase')
    plt.show()

    '''phase_shift algorithm'''
    # shift_phase = np.pi / 2
    # ref1 = reference_beam(M, N, dx, k, shift_phase=0)
    # ref2 = reference_beam(M, N, dx, k, shift_phase=shift_phase)
    # ref3 = reference_beam(M, N, dx, k, shift_phase=2*shift_phase)
    # ref4 = reference_beam(M, N, dx, k, shift_phase=3*shift_phase)
    #
    # hologram1 = interfence(I_filter, ref1)
    # hologram2 = interfence(I_filter, ref2)
    # hologram3 = interfence(I_filter, ref3)
    # hologram4 = interfence(I_filter, ref4)
    #
    # # U_complex = four_step_phase_shift(hologram1, hologram2, hologram3, hologram4, ref1)
    # U_complex = three_step_phase_shift(hologram1, hologram2, hologram3, ref1)
    #
    # U_amplitude = np.abs(U_complex)
    # U_phase = np.angle(U_complex)
    # plt.figure()
    # plt.subplot(121), plt.imshow(np.abs(U_amplitude), 'gray'), plt.axis('off'), plt.title('Amplitude')
    # plt.subplot(122), plt.imshow(np.abs(U_phase), 'gray'), plt.axis('off'), plt.title('Phase')
    # plt.show()

