ermarkedImg",watermarkedImg)
    # psnrval = psnr(coverImg,watermarkedImg)
    # print(psnrval)

    # coeffWM = pywt.dwt2(watermarkedImg,"haar")
    # hA , (hH , hV , hD) = coeffWM
    
    # coeffWM2 = pywt.dwt2(hA,"haar")
    # hA2 ,(hH2 , hV2 , hD2) = coeffWM2

    # UWM , DWM , VTWM = np.linalg.svd(hA2)
    # DE = DWM - a_c*D
    # eA = (UW @ np.diag(DE) @ VTW)

    # # extracted = (hA-a_c*cA)/a_w
    # extracted = eA , (cWH , cWV , cWD)
    # extracted = pywt.idwt2(extracted,"haar")
    # extracted *= 255
    # extracted = np.uint8(extracted)

    # cv.imshow("Extracted watermarkImg",extracted)

    # nccval = ncc(watermarkImg,extracted)
    # print(nccval)