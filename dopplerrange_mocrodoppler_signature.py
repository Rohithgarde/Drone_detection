import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pymatreader import read_mat
import pandas as pd
import math
import scipy as sp
from scipy import signal

B = 135e6 # Sweep Bandwidth
T = 36.5e-6 # Sweep Time
N = 256 # Sample Length
L = 128 
c = 3e8 # Speed ​​of Light
f0 = 77e9 # Start Frequency

NumRangeFFT = 256 # Range FFT Length
NumDopplerFFT = 128 # Doppler FFT Length

data = read_mat('data.mat')
print(data.keys())

sig = pd.DataFrame(data['C_1'])
sigReceive=sig.to_numpy()
print(sigReceive.shape)


sigRangeFFT = np.zeros((N, L), dtype=complex)
print(sigRangeFFT.shape)
sigRangeFFT=np.abs(np.fft.fft(sigReceive,128))
m=np.amax(sigRangeFFT)
print(m)
sigRangeFFT=np.divide(sigRangeFFT,m)

print(sigRangeFFT.shape)

plt.plot(sigRangeFFT)
plt.show()

sigDopplerFFT = np.zeros((N, L), dtype=complex)
sigDopplerFFT = np.fft.fft2(sigReceive)
# sigDopplerFFT = np.zeros((N, L), dtype=complex)
# for n in range(0, L):
#     sigDopplerFFT[:, n] = np.fft.fft(sigRangeFFT[:, n], NumRangeFFT)

    
print(sigDopplerFFT.shape)
sigDopplerFFT = np.fft.fftshift(sigDopplerFFT )

sigDopplerFFT = np.abs(sigDopplerFFT )
normal=sigDopplerFFT
sigDopplerFFT = 10*np.log10(sigDopplerFFT )
fig = plt.figure()
ax = Axes3D(fig)

x = np.arange(-128/2,128/2)
y = np.arange(-256/2,256/2)
X, Y = np.meshgrid(x, y)
Z = sigDopplerFFT

# U = np.abs(sigRangeFFT)

ax.plot_surface(X, Y, Z,
                rstride=2, # rstride (row) specifies the span of the row
                cstride=2, # cstride(column) specifies the span of the column
                cmap=plt.get_cmap('rainbow'))
ax.set_xlabel('doppler')
ax.set_ylabel('range')
ax.set_zlabel('Amplitude')
plt.show()
RDM=sigDopplerFFT


f, t, Zxx = sp.signal.stft((RDM), 10e3, nperseg=256)
#print("stft",Zxx)
Zxx = np.abs(Zxx)
print(Zxx)
Zxx = 20*np.log10(Zxx)
plt.imshow((Zxx),cmap='jet',interpolation='nearest', aspect='auto')
#plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()


#print(RDM)
for i in range(256):
    RDM[i][64]=0
    

Tr = 10 
Td = 8

Gr = 4
Gd = 4
Nr = 256
Nd = 128

SNR_OFFSET = 11

Gaurd_N = (2 * Gr + 1) * (2 * Gd + 1) - 1 ; 
Train_N = (2 * Tr + 2 * Gr + 1) * (2 * Td + 2 * Gd + 1) - (Gaurd_N + 1)

noise_level = np.zeros([1,1],dtype=complex)
   
for l in range(int(Tr + Gr + 1),int((Nr)) -(Tr+Gr)):
    for doppler in range(int(Td + Gd + 1),int(Nd - (Td+Gd))):
    
        noise_level = np.zeros([1,1],dtype=complex)
        
        for i in  range(int(l - (Tr + Gr)) ,int( l + Tr + Gr)): 
            for j in  range(abs(doppler - (Td + Gd)) , abs(doppler + Td + Gd)): 
                if (abs(l-i) > Gr or abs(l-j) > Gd):
                    noise_level = noise_level + normal[i][j]
        
        
        threshold = SNR_OFFSET + 10*math.log10(abs(noise_level /(2 * (Td + Gd + 1) * 2 * (Tr + Gr + 1) - (Gr * Gd) - 1)))
        
        if (RDM[l][doppler] < threshold):
            RDM[l][doppler] = 0
        else:
            RDM[l][doppler] = 1
            
            
for j in range(Nd):
    for i in range(Nr):
        if(RDM[i][j]!=0 and RDM[i][j]!=1):
            RDM[i][j]=0

            
fig = plt.figure()
ax = Axes3D(fig)
            
b = np.arange(-128/2,128/2)
x = np.multiply(b,0.2668)
a = np.arange(-256,0)
y = np.multiply(a,-0.36)
X, Y = np.meshgrid(x, y)
Z = RDM
#  print(RDM)
# U = np.abs(sigRangeFFT)
ax.plot_surface(X, Y, Z,
                rstride=2, # rstride (row) specifies the span of the row
                cstride=2, # cstride(column) specifies the span of the column
                cmap=plt.get_cmap('rainbow'))
ax.set_xlabel('doppler')
ax.set_ylabel('range')
ax.set_zlabel('Amplitude')


plt.show()
#  print(sigDopplerFFT)       

#------------stft
# f, t, Zxx = sp.signal.stft((sigDopplerFFT), 10e3, nperseg=256)
# #print("stft",Zxx)
# Zxx = np.abs(Zxx)
# print(Zxx)
# Zxx = 20*np.log10(Zxx)
# plt.imshow((Zxx),cmap='jet',interpolation='nearest', aspect='auto')
# #plt.title('STFT Magnitude')
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.show()

