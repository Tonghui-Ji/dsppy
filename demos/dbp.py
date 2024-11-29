import matplotlib.pyplot as plt
import numpy as np

from optic.dsp.core import pulseShape, firFilter, decimate, symbolSync, pnorm, signal_power, phaseNoise
from optic.models.devices import pdmCoherentReceiver


from optic.models.channels import manakovSSF, manakovDBP

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

from IPython.core.pylabtools import figsize


figsize(10, 3)

def tx_dsp(Pch_dBm=0,num_span=10):
    # Transmitter parameters:
    paramTx = parameters()
    paramTx.M   = 16           # order of the modulation format
    paramTx.Rs  = 92e9         # symbol rate [baud]
    paramTx.SpS = 16           # samples per symbol
    paramTx.pulse = 'rrc'      # pulse shaping filter
    paramTx.Ntaps = 4096     # number of pulse shaping filter coefficients
    paramTx.alphaRRC = 0.1    # RRC rolloff
    paramTx.Pch_dBm = Pch_dBm       # power per WDM channel [dBm]
    paramTx.Nch     = 1       # number of WDM channels
    paramTx.Fc      = 193.1e12 # central optical frequency of the WDM spectrum
    paramTx.lw      = 0e3    # laser linewidth in Hz
    paramTx.freqSpac = 100e9  # WDM grid spacing
    paramTx.Nmodes = 2         # number of signal modes [2 for polarization multiplexed signals]
    paramTx.Nbits = int(np.log2(paramTx.M)*1e5) # total number of bits per polarization

    # generate WDM signal
    sigWDM_Tx, symbTx_, paramTx = simpleWDMTx(paramTx)

    # optical channel parameters
    paramCh = parameters()
    paramCh.Lspan  = 50      # span length [km]
    paramCh.Ltotal = num_span*paramCh.Lspan
    paramCh.alpha = 0.2      # fiber loss parameter [dB/km]
    paramCh.D = 16           # fiber dispersion parameter [ps/nm/km]
    paramCh.gamma = 1.3      # fiber nonlinear parameter [1/(W.km)]
    paramCh.Fc = paramTx.Fc  # central optical frequency of the WDM spectrum
    paramCh.hz = 0.5         # step-size of the split-step Fourier method [km]
    paramCh.maxIter = 10      # maximum number of convergence iterations per step
    paramCh.tol = 1e-5       # error tolerance per step
    paramCh.nlprMethod = True # use adaptive step-size based o maximum nonlinear phase-shift?
    paramCh.maxNlinPhaseRot = 2e-2 # maximum nonlinear phase-shift per step
    paramCh.prgsBar = False   # show progress bar?
    paramCh.Fs = paramTx.Rs*paramTx.SpS # sampling rate
    #paramCh.saveSpanN = [1, 5, 9, 14]
    Fs = paramTx.Rs*paramTx.SpS # sampling rate
    # nonlinear signal propagation

    Pch = dBm2W(paramTx.Pch_dBm)
    sigWDM_Tx = np.sqrt(Pch/2)*pnorm(sigWDM_Tx)
    sigWDM = manakovSSF(sigWDM_Tx, paramCh)
    symbTx = symbTx_[:,:,0]
    
    print('Fiber launch power per WDM channel: ', round(10*np.log10(signal_power(sigWDM)/paramTx.Nch /1e-3),2),'dBm')
    
    return symbTx, sigWDM, paramTx, paramCh
def rx_dsp(symbTx, sigWDM, paramTx, paramCh, runDBP):
    Fs = paramTx.Rs*paramTx.SpS # sampling rate
    Fc = paramCh.Fc
    # Matched filtering
    if paramTx.pulse == 'nrz':
        pulse = pulseShape('nrz', paramTx.SpS)
    elif paramTx.pulse == 'rrc':
        pulse = pulseShape('rrc', paramTx.SpS, N=paramTx.Ntaps, alpha=paramTx.alphaRRC, Ts=1/paramTx.Rs)

    pulse = pnorm(pulse)
    sigRx = firFilter(pulse, sigWDM)  
    # DBP parameters
    paramDBP = deepcopy(paramCh)
    paramDBP.nlprMethod = False
    paramDBP.hz = paramCh.Lspan

    # CD compensation/digital backpropagation
    if runDBP:
        Pch = dBm2W(paramTx.Pch_dBm)
        sigRx = np.sqrt(Pch/2)*pnorm(sigRx)
        print('channel input power (DBP): ', round(10*np.log10(signal_power(sigRx)/1e-3),3),'dBm')

        sigRx = manakovDBP(sigRx, paramDBP)    
    else:
        paramEDC = parameters()
        paramEDC.L = paramCh.Ltotal
        paramEDC.D = paramCh.D
        paramEDC.Fc = Fc
        paramEDC.Fs = Fs
        sigRx = edc(sigRx, paramEDC)

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
    paramEq.M = paramTx.M
    paramEq.L = [int(0.2*d.shape[0]), int(0.8*d.shape[0])]
    paramEq.prgsBar = False

    if paramTx.M == 4:
        paramEq.alg = ['cma','cma'] # QPSK
        paramEq.mu = [5e-3, 1e-3] 
    else:
        paramEq.alg = ['da-rde','rde'] # M-QAM
        paramEq.mu = [5e-3, 2e-4] 

    y_EQ = mimoAdaptEqualizer(x, paramEq, d)

    ### Carrier phase recovery

    paramCPR = parameters()
    paramCPR.alg = 'bps'
    paramCPR.M   = paramTx.M
    paramCPR.N   = 75
    paramCPR.B   = 64
    
    y_CPR = cpr(y_EQ, paramCPR)
    
    discard = 5000

    ### Evaluate transmission metrics

    ind = np.arange(discard, d.shape[0]-discard)

        
    BER, SER, SNR = fastBERcalc(y_CPR[ind,:], d[ind,:], paramTx.M, 'qam')

    return np.mean(SNR)

if __name__ == '__main__':
    launch_power = np.arange(0,4,2)
    num_span = 10
    for launch_power in launch_power:
        symbTx, sigWDM, paramTx, paramCh = tx_dsp(launch_power,num_span)
        snr_wo_nlc = rx_dsp(symbTx, sigWDM, paramTx, paramCh, False)
        snr_w_dbp = rx_dsp(symbTx, sigWDM, paramTx, paramCh, True)

        print('snr wo nlc:'+ str(snr_wo_nlc))
        print('snr w dbp:'+str(snr_w_dbp))