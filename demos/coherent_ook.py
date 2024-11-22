import matplotlib.pyplot as plt
import numpy as np

from optic.dsp.core import pulseShape, firFilter, decimate, symbolSync, pnorm, signal_power, phaseNoise
from optic.models.devices import pdmCoherentReceiver, coherentReceiver

try:
    from optic.models.modelsGPU import manakovSSF, manakovDBP
except:
    from optic.models.channels import manakovSSF,linearFiberChannel

from optic.models.tx import simpleWDMTx
from optic.utils import parameters, dBm2W, dB2lin
from optic.dsp.equalization import edc, mimoAdaptEqualizer
from optic.dsp.carrierRecovery import cpr
from optic.comm.metrics import fastBERcalc, monteCarloGMI, monteCarloMI, calcEVM
from optic.plot import pconst, plotPSD

import scipy.constants as const

import logging as logg
logg.basicConfig(level=logg.WARN, format='%(message)s', force=True)

from copy import deepcopy
from tqdm.notebook import tqdm

def multipath_channel(tx_wmf, num_paths, sample_rate, phase_noise_std=np.pi/4,max_delay=10e-12):
    """
    模拟多径传输信道，并为每个路径添加独立的相位噪声。

    参数:
    - tx_wmf: 输入信号 (numpy array)
    - num_paths: 多径数量 (int)
    - sample_rate: 采样率 (Hz)
    - phase_noise_std: 相位噪声的标准差 (rad)，控制噪声强度
    
    返回:
    - rx_wmf: 经过多径和相位噪声后的接收信号 (numpy array)
    - H: 信道响应 (numpy array)
    """
    num_symbols = len(tx_wmf)
    input_freq = np.fft.fft(tx_wmf)
    frequencies = np.fft.fftfreq(num_symbols, d=1/sample_rate)

    # 初始化接收信号
    received_signal_freq = np.zeros_like(input_freq, dtype=np.complex128)

    for i in range(num_paths):
        # 每个路径独立的延迟和增益
        delay = 1e-12 * np.random.uniform(0, max_delay*1e12, size=1)  # 随机延迟
        gain = np.random.rand(1)  # 随机增益

        # 每个路径独立的相位噪声
        phase_noise = np.random.normal(0, phase_noise_std, num_symbols)
        phase_noise_factor = np.exp(1j * phase_noise)

        # 计算延迟影响并添加相位噪声
        delay_factor = np.exp(-1j * 2 * np.pi * frequencies * delay)
        received_signal_freq += gain * input_freq * delay_factor * phase_noise_factor

    # 进行逆傅里叶变换得到时域信号
    rx_wmf = np.fft.ifft(received_signal_freq)

    # 计算频域信道响应
    freq_tx = np.fft.fft(tx_wmf)
    freq_rx = np.fft.fft(rx_wmf)
    H = freq_rx / freq_tx

    # 绘制信道响应频谱
    N = len(tx_wmf)
    freq_axis = np.fft.fftfreq(N, 1/sample_rate)
    H_shifted = np.fft.fftshift(H)
    freq_axis_shifted = np.fft.fftshift(freq_axis)

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(freq_axis_shifted, np.abs(H_shifted), label='Channel Response Spectrum', color='green')
    plt.title('Channel Response Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid()
    plt.xlim(-sample_rate/2, sample_rate/2)

    plt.subplot(2, 1, 2)
    plt.plot(freq_axis_shifted, np.angle(H_shifted), label='Channel Response Phase', color='green')
    plt.title('Channel Response Phase')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Angle')
    plt.grid()
    plt.xlim(-sample_rate/2, sample_rate/2)

    plt.tight_layout()

    return rx_wmf

# Transmitter parameters:
paramTx = parameters()
paramTx.constType = 'ook'
paramTx.M   = 2           # order of the modulation format
paramTx.Rs  = 10e9         # symbol rate [baud]
paramTx.SpS = 16           # samples per symbol
paramTx.pulse = 'rect'      # pulse shaping filter
paramTx.Ntaps = 2*4096     # number of pulse shaping filter coefficients
paramTx.alphaRRC = 0.01    # RRC rolloff
paramTx.Pch_dBm = 0        # power per WDM channel [dBm]
paramTx.Nch     = 1       # number of WDM channels
paramTx.Fc      = 193.1e12 # central optical frequency of the WDM spectrum
paramTx.lw      = 0e3    # laser linewidth in Hz
paramTx.freqSpac = 37.5e9  # WDM grid spacing
paramTx.Nmodes = 1         # number of signal modes [2 for polarization multiplexed signals]
paramTx.Nbits = int(np.log2(paramTx.M)*1e5) # total number of bits per polarization

# generate WDM signal
sigWDM_Tx, symbTx_, paramTx = simpleWDMTx(paramTx)

# optical channel parameters
paramCh = parameters()
paramCh.Ltotal = 50     # total link distance [km]
paramCh.Lspan  = 50      # span length [km]
paramCh.alpha = 0.2      # fiber loss parameter [dB/km]
paramCh.D = 16           # fiber dispersion parameter [ps/nm/km]
paramCh.gamma = 1.3      # fiber nonlinear parameter [1/(W.km)]
paramCh.Fc = paramTx.Fc  # central optical frequency of the WDM spectrum
paramCh.hz = 0.5         # step-size of the split-step Fourier method [km]
paramCh.maxIter = 5      # maximum number of convergence iterations per step
paramCh.tol = 1e-5       # error tolerance per step
paramCh.nlprMethod = True # use adaptive step-size based o maximum nonlinear phase-shift?
paramCh.maxNlinPhaseRot = 2e-2 # maximum nonlinear phase-shift per step
paramCh.prgsBar = False   # show progress bar?
paramCh.Fs = paramTx.Rs*paramTx.SpS # sampling rate
Fs = paramTx.Rs*paramTx.SpS # sampling rate

### Receiver parameters

Fc = paramCh.Fc
Ts = 1/Fs
freqGrid = paramTx.freqGrid
    
## LO parameters
FO      = 0e6                 # frequency offset
lw      = 0e3                 # linewidth
Plo_dBm = 10                    # power in dBm
Plo     = dBm2W(Plo_dBm)        # power in W
ϕ_lo    = 0                     # initial phase in rad    

## photodiodes parameters
paramPD = parameters()
paramPD.B = paramTx.Rs
paramPD.Fs = Fs    
paramPD.ideal = True
    


### WDM channels coherent detection and demodulation
sigWDM = sigWDM_Tx[:,0]
sigWDM = multipath_channel(sigWDM, 20, Fs, phase_noise_std=np.pi/20,max_delay=1/paramTx.Rs)
### Receiver

# parameters
chIndex  = int(np.floor(paramTx.Nch/2))      # index of the channel to be demodulated

symbTx = symbTx_[:,:,chIndex]

#  set local oscillator (LO) parameters:   
Δf_lo   = freqGrid[chIndex]+FO  # downshift of the channel to be demodulated

# generate LO field
π       = np.pi
t       = np.arange(0, len(sigWDM))*Ts
ϕ_pn_lo = 0
sigLO   = np.sqrt(Plo)*np.exp(1j*(2*π*Δf_lo*t + ϕ_lo + ϕ_pn_lo))

#### polarization multiplexed coherent optical receiver
sigRx_coh = coherentReceiver(sigWDM, sigLO, paramPD)

### Matched filtering and CD compensation

# Rx filtering

# Matched filtering
if paramTx.pulse == 'nrz':
    pulse = pulseShape('nrz', paramTx.SpS)
elif paramTx.pulse == 'rrc':
    pulse = pulseShape('rrc', paramTx.SpS, N=paramTx.Ntaps, alpha=paramTx.alphaRRC, Ts=1/paramTx.Rs)
elif paramTx.pulse == 'rect':
    pulse = pulseShape('rect', paramTx.SpS)

pulse = pnorm(pulse)
sigRx = firFilter(pulse, sigRx_coh)  
sigRx = sigRx_coh

### Downsampling to 2 samples/symbol and re-synchronization with transmitted sequences

# decimation
paramDec = parameters()
paramDec.SpS_in  = paramTx.SpS
paramDec.SpS_out = 2
sigRx = decimate(sigRx, paramDec)

symbRx = symbolSync(sigRx, symbTx, 2)

### Power normalization

x = pnorm(sigRx)
d = pnorm(symbRx)

### Adaptive equalization

# adaptive equalization parameters
paramEq = parameters()
paramEq.nTaps = 15
paramEq.SpS = paramDec.SpS_out
paramEq.numIter = 5
paramEq.storeCoeff = False
paramEq.constType = paramTx.constType
paramEq.M = paramTx.M
# paramEq.L = [int(0.2*d.shape[0]), int(0.8*d.shape[0])]
paramEq.prgsBar = False


# paramEq.alg = ['da-rde','rde'] # M-QAM
# paramEq.mu = [5e-3, 2e-4] 
paramEq.alg = 'dd-lms'
paramEq.mu = 2e-4
y_wo_EQ = x[0:-1:2]
y_EQ = mimoAdaptEqualizer(x, paramEq, d)

### Carrier phase recovery

paramCPR = parameters()
paramCPR.alg = 'bps'
paramCPR.constType = paramTx.constType
paramCPR.M   = paramTx.M
paramCPR.N   = 75
paramCPR.B   = 64

# y_CPR = cpr(y_EQ, paramCPR)
pconst(y_wo_EQ,R=2)
pconst(y_EQ,R=2)
discard = 5000

### Evaluate transmission metrics

ind = np.arange(discard, d.shape[0]-discard)
    
BER, SER, SNR = fastBERcalc(y_EQ[ind,:], d[ind,:], paramTx.M, paramTx.constType)
print(f'BER is: {BER}')
plt.show()
pass
