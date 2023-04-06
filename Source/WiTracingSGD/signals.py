import numpy as np

def dBmTomW(dBm):
    return np.power(10.0, dBm / 10.0)

def mWTodBm(mW):
    return 10*np.log10(mW)

def calc_rss_at(pt_mW, d):
    return pt_mW / (4 * np.pi * d * d)

def calc_distance_with(pt_mW, rss_dBm):
    return np.sqrt(pt_mW / (dBmTomW(rss_dBm) * 4 * np.pi))


if __name__ == "__main__":
    print("-30 dBm to mW:",dBmTomW(-30))
    print("0.001 mW to dBm:",mWTodBm(0.001))
    print("RSS at distance 20m with 0.001mW transmit power", int(mWTodBm(calc_rss_at(0.001, 20))))

