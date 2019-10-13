from Holography_Simulation import *


def low_filter_square(img, short_edge, xm, ym):
    f_img = np.fft.fftshift(np.fft.fft2(img))
    [M, N] = np.shape(img)
    long_edge = int(np.round(short_edge * (N / M)))
    f_filter = f_img[ym-short_edge:ym+short_edge, xm-long_edge:xm+long_edge]
    return np.fft.ifft2(np.fft.ifftshift(f_filter))


def PCA_Aberration(I, order):
    [M, N] = np.shape(I)
    rfi = np.exp(-1j * np.angle(I))
    [u, s, v] = np.linalg.svd(rfi)      # SVD分解

    ss = np.zeros(np.shape(I))
    ss[0, 0] = s[0]
    unpoly = np.dot(np.dot(np.mat(u), np.mat(ss)), np.mat(v))   # 未拟合PCA

    # 拟合
    unu = np.unwrap(np.angle(u[:, 0]))
    A = np.polyfit(range(M), unu.transpose(), order)
    newUNU = np.polyval(A, range(M))
    newU = np.exp(1j * newUNU.transpose())
    u[:, 0] = newU

    unv = np.unwrap(np.angle(v[:, 0]))
    A = np.polyfit(range(N), unv.transpose(), order)
    newUNV = np.polyval(A, range(N))
    newV = np.exp(1j * newUNV.transpose())
    v[:, 0] = newV

    aberration_conj = np.dot(np.dot(np.mat(u), np.mat(ss)), np.mat(v))  # 相差共轭项
    I_p = np.asarray(I) * np.asarray(aberration_conj)                   # 相差补偿

    plt.figure()
    plt.subplot(221), plt.imshow(np.angle(I), 'gray'), plt.axis('off'), plt.title('raw phase')
    plt.subplot(222), plt.imshow(np.angle(unpoly), 'gray'), plt.axis('off'), plt.title('unfitted phase')
    plt.subplot(223), plt.imshow(np.angle(aberration_conj), 'gray'), plt.axis('off'), plt.title('conjugated phase')
    plt.subplot(224), plt.imshow(np.angle(I_p), 'gray'), plt.axis('off'), plt.title('Aberration Compensation')
    # plt.show()

    return I_p


def spectrum_padding(I, M, N, dx, dy):
    f_I = np.fft.fftshift(np.fft.fft2(I))
    f_padding = np.zeros((M, N), dtype=complex)
    f_padding[int(M/2 - dx):int(M/2 + dx), int(N/2 - dy):int(N/2 + dy)] = f_I
    return np.fft.ifft2(np.fft.ifftshift(f_padding))


if __name__ == "__main__":
    img = cv2.imread('Imgs/UASF_Experiment.bmp', 0)
    [M, N] = np.shape(img)

    aperture = 120
    [xm, ym, dx, dy] = order_location(img, degree_x=150, degree_y=0)
    hologram_filter = low_filter_square(img, aperture, xm, ym)

    # plt.figure()
    # plt.subplot(121), plt.imshow(np.abs(hologram_filter), 'gray'), plt.axis('off')
    # plt.subplot(122), plt.imshow(np.angle(hologram_filter), 'gray'), plt.axis('off')
    # plt.show()

    hologram_compensation = PCA_Aberration(hologram_filter, 2)
    plt.figure()
    plt.subplot(121), plt.imshow(np.abs(hologram_compensation), 'gray'), plt.axis('off')
    plt.subplot(122), plt.imshow(np.angle(hologram_compensation), 'gray'), plt.axis('off')
    # plt.show()

    hologram_compensation = spectrum_padding(hologram_compensation, M, N, aperture, int(np.round(aperture*(N / M))))
    # hologram_compensation = PCA_Aberration(hologram_compensation, 1)
    plt.figure()
    plt.subplot(121), plt.imshow(np.abs(hologram_compensation), 'gray'), plt.axis('off')
    plt.subplot(122), plt.imshow(np.angle(hologram_compensation), 'gray'), plt.axis('off')
    plt.show()
    cv2.imshow('image', 10*np.abs(hologram_compensation)/np.max(np.abs(hologram_compensation)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

