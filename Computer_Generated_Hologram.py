import numpy as np
import cv2
import matplotlib.pyplot as plt
import Diffraction


def GS_fft(img, iteration):
    '''
    GS FFT计算全息
    :param img: 图像
    :param iteration: 迭代次数
    :return: SLM
    '''

    [M, N] = np.shape(img)

    random_phi = np.random.random((M, N))*np.pi                 # 生成随机相位
    hologram_plane = np.ones((M, N))*np.exp(1j*random_phi)      # 生成初始全息面

    for iter in range(iteration):
        obj_temp = np.fft.fftshift(np.fft.fft2(hologram_plane))     # 传播到物平面
        obj_plane = img*np.exp(1j*np.angle(obj_temp))               # 保留相位，用图像替代振幅

        hologram_temp = np.fft.ifft2(np.fft.ifftshift(obj_plane))               # 传播到物平面
        hologram_plane = np.ones((M, N))*np.exp(1j*np.angle(hologram_temp))     # 保留相位，用单位振幅代替振幅

    slm = np.angle(hologram_plane)
    return slm


def GS_fresnel(img, iteration, z, dx, r):
    '''
    GS Fresnel不同距离计算全息
    :param img: 图像
    :param iteration: 迭代次数
    :param z: 衍射距离
    :param dx: 采样间隔
    :param r: 波长
    :return: SLM
    '''

    [M, N] = np.shape(img)

    random_phi = np.random.random((M, N)) * np.pi  # 生成随机相位
    hologram_plane = np.ones((M, N)) * np.exp(1j * random_phi)  # 生成初始全息面

    for iter in range(iteration):
        [obj_temp, I, dfx] = Diffraction.fresnel_diff_s_fft(hologram_plane, z, dx, r)
        obj_plane = img * np.exp(1j * np.angle(obj_temp))

        [hologram_temp, I, dx] = Diffraction.fresnel_diff_s_fft(obj_plane, -z, dfx, r)
        norm = np.sqrt(np.sum(np.abs(hologram_temp)**2)/(M*N))
        hologram_plane = norm*np.ones((M, N))*np.exp(1j*np.angle(hologram_temp))

    slm = np.angle(hologram_plane)
    return slm


def Burch(img, alpha, dx):
    '''
    伯奇编码实现计算全息
    :param img: 图像
    :param alpha: 载波
    :param dx: 采样间隔
    :return: SLM
    '''

    [M, N] = np.shape(img)

    lx = dx*N
    ly = dx*M

    x = np.linspace(-lx/2, lx/2, N)
    y = np.linspace(-ly/2, ly/2, M)
    [x, y] = np.meshgrid(x, y)

    ef = 0.5    # 随机纯相位噪声的系数
    uf = np.fft.fftshift(np.fft.fft2(img*np.exp(1j*np.random.random((M, N))*np.pi*ef)))     # 计算频谱并频移
    amplitude = np.abs(uf)                                  # 频谱振幅
    amplitude = amplitude/np.max(amplitude)                 # 归一化振幅
    phase = np.angle(uf)                                    # 频谱相位
    H = 0.5*(1+amplitude*np.cos(2*np.pi*alpha*x-phase))     # 伯奇编码
    H = np.round(H/np.max(H)*255)                           # 灰度量化
    return H


def kinoform(img):
    '''
    相息图法
    :param img:
    :return: SLM
    '''

    [M, N] = np.shape(img)
    ef = 0.5
    uf = np.fft.fftshift(np.fft.fft2(img*np.exp(1j*np.random.random((M, N))*np.pi*ef)))
    phase = np.angle(uf)
    H = np.mod(phase, 2*np.pi)
    SLM = np.round(H/np.max(H)*255)
    return SLM


if __name__ == "__main__":
    r = 0.532e-6  # 波长
    dx = 6.4e-6  # 初始面采样间隔
    z = 0.15

    # GS
    # img = cv2.imread("Imgs/num.jpg", 0)

    # plt.figure()
    # plt.imshow(img, 'gray'), plt.axis('off'), plt.title('Raw')

    # GS_FFT
    # SLM = GS_fft(img, 20)
    # U = np.fft.fftshift(np.fft.fft2(np.exp(1j*SLM)))
    # I = U*np.conj(U)

    # GS_Fresnel
    # SLM = GS_fresnel(img, 20, z, dx, r)
    # [U, I, dfx] = Diffraction.fresnel_diff_s_fft(np.exp(1j*SLM), z, dx, r)

    # Multiple_Plane
    # img1 = cv2.imread("Imgs/A.png", 0)
    # img2 = cv2.imread("Imgs/B.png", 0)
    # img3 = cv2.imread("Imgs/C.png", 0)
    # img4 = cv2.imread("Imgs/D.png", 0)
    #
    # SLM_A = GS_fresnel(img1, 20, 0.10, dx, r)
    # SLM_B = GS_fresnel(img2, 20, 0.12, dx, r)
    # SLM_C = GS_fresnel(img3, 20, 0.14, dx, r)
    # SLM_D = GS_fresnel(img4, 20, 0.16, dx, r)
    #
    # SLM = np.exp(1j*SLM_A) + np.exp(1j*SLM_B) + np.exp(1j*SLM_C) + np.exp(1j*SLM_D)
    # [U, I, dfx] = Diffraction.fresnel_diff_s_fft(SLM, 0.14, dx, r)

    # plt.figure()
    # plt.subplot(121), plt.imshow(np.abs(SLM), 'gray'), plt.axis("off"), plt.title('SLM')
    # plt.subplot(122), plt.imshow(np.abs(I), 'gray'), plt.axis("off"), plt.title('Reconstruction')
    # plt.show()

    # 其他编码方法
    img = cv2.imread("Imgs/Copy_of_A.png", 0)

    # Burch
    SLM = Burch(img, 60000, dx)
    uI = np.fft.ifft2(SLM)
    I = uI*np.conj(uI)

    # kinoform
    # SLM = kinoform(img)
    # uI = np.fft.ifft2(np.exp(1j*SLM/40.58))
    # I = uI*np.conj(uI)

    plt.figure()
    plt.subplot(121), plt.imshow(np.abs(SLM), 'gray'), plt.axis("off"), plt.title('SLM')
    plt.subplot(122), plt.imshow(np.abs(I), 'gray', vmin=0, vmax=0.001), plt.axis("off"), plt.title('Reconstruction')
    plt.show()