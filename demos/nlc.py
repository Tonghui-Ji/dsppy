import matplotlib.pyplot as plt
import numpy as np

from optic.dsp.core import pulseShape, firFilter, decimate, symbolSync, pnorm, signal_power, phaseNoise
from optic.models.devices import pdmCoherentReceiver

try:
    from optic.models.modelsGPU import manakovSSF, manakovDBP
except:
    from optic.models.channels import manakovSSF

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

# Transmitter parameters:
paramTx = parameters()
paramTx.M   = 64           # order of the modulation format
paramTx.Rs  = 32e9         # symbol rate [baud]
paramTx.SpS = 16           # samples per symbol
paramTx.pulse = 'rrc'      # pulse shaping filter
paramTx.Ntaps = 2*4096     # number of pulse shaping filter coefficients
paramTx.alphaRRC = 0.01    # RRC rolloff
paramTx.Pch_dBm = 0        # power per WDM channel [dBm]
paramTx.Nch     = 1       # number of WDM channels
paramTx.Fc      = 193.1e12 # central optical frequency of the WDM spectrum
paramTx.lw      = 100e3    # laser linewidth in Hz
paramTx.freqSpac = 37.5e9  # WDM grid spacing
paramTx.Nmodes = 2         # number of signal modes [2 for polarization multiplexed signals]
paramTx.Nbits = int(np.log2(paramTx.M)*1e5) # total number of bits per polarization

# generate WDM signal
sigWDM_Tx, symbTx_, paramTx = simpleWDMTx(paramTx)

# optical channel parameters
paramCh = parameters()
paramCh.Ltotal = 150     # total link distance [km]
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
#paramCh.saveSpanN = [1, 5, 9, 14]
Fs = paramTx.Rs*paramTx.SpS # sampling rate

# DBP parameters
paramDBP = deepcopy(paramCh)
paramDBP.nlprMethod = False
paramDBP.hz = 10
runDBP = True

### Receiver parameters

Fc = paramCh.Fc
Ts = 1/Fs
freqGrid = paramTx.freqGrid
    
## LO parameters
FO      = 150e6                 # frequency offset
lw      = 100e3                 # linewidth
Plo_dBm = 10                    # power in dBm
Plo     = dBm2W(Plo_dBm)        # power in W
ϕ_lo    = 0                     # initial phase in rad    

## photodiodes parameters
paramPD = parameters()
paramPD.B = paramTx.Rs
paramPD.Fs = Fs    
paramPD.ideal = True
    
Powers = paramTx.Pch_dBm + np.arange(-8,0,0.5)
scale = np.arange(0,0.5,0.5)

BER = np.zeros((4,len(Powers)))
SER = np.zeros((4,len(Powers)))
MI  = np.zeros((4,len(Powers)))
GMI = np.zeros((4,len(Powers)))
NGMI = np.zeros((4,len(Powers)))
SNR = np.zeros((4,len(Powers)))
EVM = np.zeros((4,len(Powers)))

for indP, G in enumerate(tqdm(scale)):
    # nonlinear signal propagation
    G_lin = dB2lin(G)

    sigWDM = manakovSSF(np.sqrt(G_lin)*sigWDM_Tx, paramCh)
    print('Fiber launch power per WDM channel: ', round(10*np.log10(signal_power(sigWDM)/paramTx.Nch /1e-3),2),'dBm')
    
    ### WDM channels coherent detection and demodulation

    ### Receiver

    # parameters
    chIndex  = int(np.floor(paramTx.Nch/2))      # index of the channel to be demodulated

#     print('Demodulating channel #%d , fc: %.4f THz, λ: %.4f nm\n'\
#           %(chIndex, (Fc + freqGrid[chIndex])/1e12, const.c/(Fc + freqGrid[chIndex])/1e-9))

    symbTx = symbTx_[:,:,chIndex]

    #  set local oscillator (LO) parameters:   
    Δf_lo   = freqGrid[chIndex]+FO  # downshift of the channel to be demodulated

    # generate LO field
    π       = np.pi
    t       = np.arange(0, len(sigWDM))*Ts
    ϕ_pn_lo = phaseNoise(lw, len(sigWDM), Ts)
    sigLO   = np.sqrt(Plo)*np.exp(1j*(2*π*Δf_lo*t + ϕ_lo + ϕ_pn_lo))

    #### polarization multiplexed coherent optical receiver
    θsig = π/3 # polarization rotation angle
    sigRx_coh = pdmCoherentReceiver(sigWDM, sigLO, θsig, paramPD)


    ### Matched filtering and CD compensation
    
    # Rx filtering

    # Matched filtering
    if paramTx.pulse == 'nrz':
        pulse = pulseShape('nrz', paramTx.SpS)
    elif paramTx.pulse == 'rrc':
        pulse = pulseShape('rrc', paramTx.SpS, N=paramTx.Ntaps, alpha=paramTx.alphaRRC, Ts=1/paramTx.Rs)

    pulse = pnorm(pulse)
    sigRx = firFilter(pulse, sigRx_coh)  

    Pch = dBm2W(G + paramTx.Pch_dBm)
    sigRx = np.sqrt(Pch/2)*pnorm(sigRx)
    #print('channel input power (DBP): ', round(10*np.log10(signal_power(sigRx)/1e-3),3),'dBm')

    sigRx,_ = manakovDBP(sigRx, Fs, paramDBP)    


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

    # remove phase and polarization ambiguities for QPSK signals
    if paramTx.M == 4:   
        d = symbTx
        # find rotations after CPR and/or polarizations swaps possibly added at the output the adaptive equalizer:
        rot0 = [np.mean(pnorm(symbTx[ind,0])/pnorm(y_CPR[ind,0])), np.mean(pnorm(symbTx[ind,1])/pnorm(y_CPR[ind,0]))]
        rot1 = [np.mean(pnorm(symbTx[ind,1])/pnorm(y_CPR[ind,1])), np.mean(pnorm(symbTx[ind,0])/pnorm(y_CPR[ind,1]))]

        if np.argmax(np.abs(rot0)) == 1 and np.argmax(np.abs(rot1)) == 1:      
            y_CPR_ = y_CPR.copy() 
            # undo swap and rotation 
            y_CPR[:,0] = pnorm(rot1[np.argmax(np.abs(rot1))]*y_CPR_[:,1]) 
            y_CPR[:,1] = pnorm(rot0[np.argmax(np.abs(rot0))]*y_CPR_[:,0])
        else:
            # undo rotation
            y_CPR[:,0] = pnorm(rot0[np.argmax(np.abs(rot0))]*y_CPR[:,0])
            y_CPR[:,1] = pnorm(rot1[np.argmax(np.abs(rot1))]*y_CPR[:,1])


        indsave = np.arange(0,2)

            
        BER[indsave,indP], SER[indsave,indP], SNR[indsave,indP] = fastBERcalc(y_CPR[ind,:], d[ind,:], paramTx.M, 'qam')
        GMI[indsave,indP], NGMI[indsave,indP] = monteCarloGMI(y_CPR[ind,:], d[ind,:], paramTx.M, 'qam')
        MI[indsave,indP] = monteCarloMI(y_CPR[ind,:], d[ind,:], paramTx.M, 'qam')
        EVM[indsave,indP] = calcEVM(y_CPR[ind,:], paramTx.M, 'qam', d[ind,:])

        print('      pol.X      pol.Y      ')
        print(' SER: %.2e,  %.2e'%(SER[indsave[0],indP], SER[indsave[1],indP]))
        print(' BER: %.2e,  %.2e'%(BER[indsave[0],indP], BER[indsave[1],indP]))
        print(' SNR: %.2f dB,  %.2f dB'%(SNR[indsave[0],indP], SNR[indsave[1],indP]))
        print(' EVM: %.2f %%,    %.2f %%'%(EVM[indsave[0],indP]*100, EVM[indsave[1],indP]*100))
        print('  MI: %.2f bits, %.2f bits'%(MI[indsave[0],indP], MI[indsave[1],indP]))
        print(' GMI: %.2f bits, %.2f bits'%(GMI[indsave[0],indP], GMI[indsave[1],indP]))
        print('NGMI: %.2f,      %.2f'%(NGMI[indsave[0],indP], NGMI[indsave[1],indP]))

        fig, ax = plt.subplots(1,4, layout="constrained", figsize=(18,9))

ax[0].plot(Powers, np.log10(BER.T), '-*', label=['x-pol DBP', 'y-pol DBP', 'x-pol EDC', 'y-pol EDC']);
ax[0].set_xlabel('Power [dBm]')
ax[0].set_ylabel('log10(BER)')
ax[0].legend()
ax[0].grid()
ax[0].set_box_aspect(0.75)
ax[0].set_xlim(min(Powers), max(Powers))

ax[1].plot(Powers, np.log10(SER.T), '-*', label=['x-pol DBP', 'y-pol DBP', 'x-pol EDC', 'y-pol EDC']);
ax[1].set_xlabel('Power [dBm]')
ax[1].set_ylabel('log10(SER)')
ax[1].legend()
ax[1].grid()
ax[1].set_box_aspect(0.75)
ax[1].set_xlim(min(Powers), max(Powers))

ax[2].plot(Powers, SNR.T, '-*', label=['x-pol DBP', 'y-pol DBP', 'x-pol EDC', 'y-pol EDC']);
ax[2].set_xlabel('Power [dBm]')
ax[2].set_ylabel('SNR [dB]')
ax[2].legend()
ax[2].grid()
ax[2].set_box_aspect(0.75)
ax[2].set_xlim(min(Powers), max(Powers))

ax[3].plot(Powers, GMI.T, '-*', label=['x-pol DBP', 'y-pol DBP', 'x-pol EDC', 'y-pol EDC']);
ax[3].set_xlabel('Power [dBm]')
ax[3].set_ylabel('GMI [bits]')
ax[3].legend()
ax[3].grid()
ax[3].set_box_aspect(0.75);
ax[3].set_xlim(min(Powers), max(Powers));

#fig.tight_layout()
#fig.set_size_inches(15, 20)