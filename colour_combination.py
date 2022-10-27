import numpy as np

def TPS_color_correction(input_color,actual_color,reference):
    NPs = actual.shape[0]
    NPg = reference.shape[0]
    Xp = actual[:, 0].conj().transpose()
    Yp = actual[:, 1].conj().transpose()
    Zp = actual[:, 2].conj().transpose()
    Xg = reference[:, 0].conj().transpose()
    Yg = reference[:, 1].conj().transpose()
    Zg = reference[:, 2].conj().transpose()

    rXp = np.tile(Xp.reshape(NPs, 1), (1, NPs))
    rYp = np.tile(Yp.reshape(NPs, 1), (1, NPs))
    rZp = np.tile(Zp.reshape(NPs, 1), (1, NPs))
    wR=np.sqrt(((rXp - rXp.conj().transpose()) ** 2 + (rYp - rYp.conj().transpose()) ** 2 + (rZp - rZp.conj().transpose()) ** 2))
    wK = (2 * (wR ** 2)) * np.log(wR + 1e-20)
    wP = np.hstack((np.ones((NPs, 1)), Xp.reshape(NPs, 1), Yp.reshape(NPs, 1), Zp.reshape(NPs, 1)))
    wL = np.vstack((np.hstack((wK, wP)),np.hstack((wP.conj().transpose(), np.zeros((4,4))))))
    wY = np.vstack((np.hstack((Xg.reshape(NPg, 1), Yg.reshape(NPg, 1), Zg.reshape(NPg, 1))),np.zeros((4, 3))))
    inv_wl=np.linalg.inv(wL)
    wW = np.dot(inv_wl, wY)

    X = input_color[:, 0].conj().transpose()
    Y = input_color[:, 1].conj().transpose()
    Z = input_color[:, 2].conj().transpose()
    NPss=input_color.shape[0]
    NWs = np.size(X, axis=0)
    rX = np.tile(X, (NPss,1))
    rY = np.tile(Y, (NPss,1))
    rZ = np.tile(Z, (NPss,1))
    rXp = np.tile(Xp.reshape(NPs, 1), (1,NWs))
    rYp = np.tile(Yp.reshape(NPs, 1), (1,NWs))
    rZp = np.tile(Zp.reshape(NPs, 1), (1,NWs))

    wR = np.sqrt((rXp - rX)**2 + (rYp - rY)**2 + (rZp - rZ)** 2)
    wK = (2 * (wR ** 2)) * np.log(wR + 1e-20)
    wP = np.hstack((np.ones((NWs, 1)), X.reshape(NPss,1), Y.reshape(NPss,1), Z.reshape(NPss,1))).conj().transpose()
    wL = np.vstack((wK,wP)).conj().transpose()

    Xw = np.dot(wL,wW[:, 0])
    Yw = np.dot(wL,wW[:, 1])
    Zw = np.dot(wL,wW[:, 2])
    return np.hstack((Xw.reshape(NPss,1), Yw.reshape(NPss,1), Zw.reshape(NPss,1)))
