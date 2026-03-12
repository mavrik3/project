from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
from itertools import groupby
from operator import itemgetter
import scipy

from itertools import groupby
from operator import itemgetter


class IQ:
    """Utility class for processing Bluetooth Low Energy (BLE) IQ signal frames.

    Provides signal processing operations (FFT, magnitude, phase, filtering,
    demodulation, etc.) that can be applied to raw NumPy arrays or to columns
    of a Pandas DataFrame. When a DataFrame is passed, operations are applied
    row-wise using :meth:`inputCheck`.

    Attributes:
        Warnings (bool): Whether to print warning messages for missing
            parameters. Defaults to True.
        Fc (float): Center frequency in Hz. Defaults to 2.44 GHz.
        Fs (float): Sampling frequency in Hz. Defaults to 100 MHz.
        figsize (tuple): Default figure size for plots.
        dpi (int): Default DPI for plots.
        df (pd.DataFrame | None): Optional default DataFrame to operate on.
        BLEChnls (np.ndarray): Array of BLE channel center frequencies in Hz.
        onBodyMap (dict): Mapping from position index to
            ``[body_region, side]`` labels for on-body measurements.
    """
    Fc = 2.44e9
    Fs = 100e6
    figsize = (20,3)
    dpi = 100
    df = None
    BLEChnls = np.array([2404000000,2406000000,2408000000,2410000000,
    2412000000,2414000000,2416000000,2418000000,2420000000,2422000000,
    2424000000,2428000000,2430000000,2432000000,2434000000,2436000000,
    2438000000,2440000000,2442000000,2444000000,2446000000,2448000000,      
    2450000000,2452000000,2454000000,2456000000,2458000000,2460000000,
    2462000000,2464000000,2466000000,2468000000,2470000000,2472000000,
    2474000000,2476000000,2478000000,2402000000,2426000000,2480000000])

    onBodyMap = {1: ['head','right'],              2: ['head','left'], 
                  3: ['chest', 'right'],            4: ['chest', 'left'],
                  5: ['fornTorso', 'right'],        6: ['fornTorso', 'left'],
                  7: ['arm', 'right'],              8: ['arm', 'left'],
                  9: ['wrist', 'right'],           10: ['wrist', 'left'],
                  11: ['backTorso', 'right'],      12: ['backTorso', 'left']}
    
    def __init__(self, df=None, Fc=None, Fs=None, Warnings=True):
        """Initialize the IQ processor with optional default parameters.

        Args:
            df: Default DataFrame to operate on when no explicit input is
                passed to processing methods.
            Fc: Center frequency in Hz. If None, uses the class default
                (2.44 GHz).
            Fs: Sampling frequency in Hz. If None, uses the class default
                (100 MHz).
            Warnings: Whether to print warnings for missing parameters.
        """
        if Fs is not None:
            self.Fs = Fs
        if Fc is not None:
            self.Fc = Fc
        if df is not None: 
            self.df = df
        self.Warnings = Warnings

    def isList(self, input):
        """Check whether *input* is a list or NumPy array.

        Args:
            input: Value to test.

        Returns:
            True if *input* is a ``list`` or ``np.ndarray``, False otherwise.
        """
        return isinstance(input, list) or isinstance(input, np.ndarray)
    
    def isPandaDF(self, input):
        """Check whether *input* is a Pandas DataFrame or Series.

        Args:
            input: Value to test.

        Returns:
            True if *input* is a ``pd.DataFrame`` or ``pd.Series``, False
            otherwise.
        """
        return isinstance(input, pd.DataFrame) or isinstance(input, pd.Series)

    def inputCheck(self, input, method=None, col_name=None, args=None, plot=False):
        """Route an operation to the appropriate execution path based on input type.

        Dispatches *method* to run on a raw array/list or, for DataFrames,
        applies it row-wise using :meth:`pandas.DataFrame.apply`. When
        *col_name* is given and *input* is a DataFrame the result is stored
        back in ``self.df``.

        Args:
            input: IQ data to process. Accepts a NumPy array, list,
                ``pd.Series``, or ``pd.DataFrame``. Falls back to ``self.df``
                if None.
            method: Callable that accepts a single IQ array (plus optional
                keyword arguments from *args*) and returns a processed array.
            col_name: Column name used to store results when *input* is a
                DataFrame.
            args: Optional dict of keyword arguments forwarded to *method*.
            plot: If True and *input* is a DataFrame, uses a plot-specific
                dispatch that reads ``title``, ``x_label``, and ``y_label``
                columns.

        Returns:
            Processed result (NumPy array, scalar, Series, or DataFrame
            depending on *input* type and *col_name*).
        """
        if input is None:
            if self.df is None:
                print("error: no input")
            else:
                input = self.df
        if method is None:
            print("error: no method")
            return
        

        if self.isList(input):
            if args is not None:
                return method(input, **args)
            else:
                return method(input)
        
        elif self.isPandaDF(input):
            if isinstance(input, pd.Series):
                if args is not None:
                    res = input.apply(lambda x: method(x,**args))
                else:
                    res = input.apply(lambda x: method(x))
                
            elif plot: # bad way to handle plot but this is a quick fix 
                try:  
                    res = input.apply(lambda x: method(x[col_name],x['title'],x['x_label'],x['y_label'], x['x']) , axis=1)
                except:
                    print("Warning: No x/y_label columns")
                    try:
                        res = input.apply(lambda x: method(x[col_name],x['title']) , axis=1)
                    except:
                        if self.Warnings:
                            print("Warning: Np title columns")
                        res = input.apply(lambda x: method(x[col_name],**args) , axis=1)

                return True 
            
            elif 'frame' in input.columns:
                if args is not None:
                    res = input.apply(lambda x: method(x['frame'],**args) , axis=1)
                else:
                    res = input.apply(lambda x: method(x['frame']) , axis=1)
            elif 'I' in input.columns and 'Q' in input.columns:
                res = input.apply(lambda x: method(x['I'] + np.dot(x['Q'],1j),**args) , axis=1)
            else:  
                print("error: input does not contain frame or I/Q columns")

            if col_name is not None:
                self.df[col_name] = res
                return self.df
            else:
                return res


    def _abs(self, input):
        return np.abs(input)

    def abs(self, frame: np.ndarray | pd.DataFrame = None, col_name=None):
        """Compute the magnitude (absolute value) of a complex IQ signal.

        Args:
            frame: Input IQ data as a NumPy array or DataFrame. Falls back to
                ``self.df`` if None.
            col_name: If provided and *frame* is a DataFrame, stores the result
                in this column of ``self.df``.

        Returns:
            Magnitude array or updated DataFrame.
        """
        return self.inputCheck(frame, method=self._abs, col_name=col_name)

    def _phase(self, input):
        return np.angle(input)

    def phase(self, frame: np.ndarray | pd.DataFrame = None, col_name=None):
        """Compute the instantaneous phase of a complex IQ signal.

        Args:
            frame: Input IQ data as a NumPy array or DataFrame.
            col_name: Optional column name for in-place DataFrame storage.

        Returns:
            Phase array (in radians) or updated DataFrame.
        """
        return self.inputCheck(frame, method=self._phase, col_name=col_name)

    def _fft(self, input):
        return np.fft.fftshift(np.fft.fft(input))

    def fft(self, frame: np.ndarray | pd.DataFrame = None, col_name=None):
        """Compute the centered FFT of an IQ signal.

        Args:
            frame: Input IQ data as a NumPy array or DataFrame.
            col_name: Optional column name for in-place DataFrame storage.

        Returns:
            Frequency-domain array (zero-centered) or updated DataFrame.
        """
        return self.inputCheck(frame, method=self._fft, col_name=col_name)

    def _shift(self, input, shift=0):
        return input * np.exp(2j * np.pi * shift * np.linspace(0, len(input), len(input)) / len(input))

    def shift(self, frame: np.ndarray | pd.DataFrame = None, shift=0, col_name=None):
        """Frequency-shift an IQ signal by a given offset.

        Args:
            frame: Input IQ data as a NumPy array or DataFrame.
            shift: Frequency shift to apply (normalized to sample rate).
            col_name: Optional column name for in-place DataFrame storage.

        Returns:
            Frequency-shifted signal or updated DataFrame.
        """
        return self.inputCheck(frame, method=self._shift, col_name=col_name, args={"shift": shift})
    
    def _rssi(self, input):
        input = input[100:-100]
        return 10 * np.log(np.average(np.sqrt(np.imag(input) ** 2 + np.real(input) ** 2)))

    def rssi(self, frame: np.ndarray | pd.DataFrame = None, col_name=None):
        """Estimate the Received Signal Strength Indicator (RSSI) in dB.

        Trims 100 samples from each end of the frame to avoid transient
        effects, then computes 10 * log(mean(|signal|)).

        Args:
            frame: Input IQ data as a NumPy array or DataFrame.
            col_name: Optional column name for in-place DataFrame storage.

        Returns:
            RSSI value in dB, or a Series/DataFrame of values if *frame* is
            a DataFrame.
        """
        return self.inputCheck(frame, method=self._rssi, col_name=col_name)

    def _channelDetection(self, input, Fs=Fs):
        fft = self.fft(input)
        absfft = np.abs(fft)
        n0 = np.where(absfft == np.max(absfft))[0][0]
        f = np.arange(-self.Fs / 2, Fs / 2, Fs / len(absfft))
        c0 = f[n0] + self.Fc
        try:
            return np.where(abs(self.BLEChnls - c0) < 1e6)[0][0]
        except Exception:
            return -1

    def channelDetection(self, frame: np.ndarray | pd.DataFrame = None, col_name=None, Fs=None):
        """Detect the BLE channel index of an IQ frame.

        Identifies the dominant frequency component via FFT and maps it to
        the nearest BLE channel center frequency.

        Args:
            frame: Input IQ data as a NumPy array or DataFrame.
            col_name: Optional column name for in-place DataFrame storage.
            Fs: Sampling frequency in Hz. Defaults to ``self.Fs`` with a
                warning if not specified.

        Returns:
            BLE channel index (0-based), or -1 if no matching channel is
            found. Returns a Series/DataFrame of indices when *frame* is a
            DataFrame.
        """
        if Fs is None:
            Fs = self.Fs
            if self.Warnings:
                print("Warning: (channelDetection) No sampling frequency specified, using default Fs of {}Msps.".format(Fs / 1e6))
        return self.inputCheck(frame, method=self._channelDetection, col_name=col_name, args={"Fs": Fs})

    def _demodulate(self, input, Fs=None):
        chnl = self._channelDetection(input, Fs=Fs)
        Fc = self.BLEChnls[chnl]
        diffFc = (self.Fc - Fc) / (Fs / len(input))
        return input * np.exp(2j * np.pi * diffFc * np.linspace(0, len(input), len(input)) / len(input))

    def demodulate(self, frame: np.ndarray | pd.DataFrame = None, col_name=None, Fs=None):
        """Frequency-demodulate a BLE IQ frame to the detected channel center.

        Detects the channel via :meth:`channelDetection`, then applies a
        complex frequency shift to bring the signal to baseband.

        Args:
            frame: Input IQ data as a NumPy array or DataFrame.
            col_name: Optional column name for in-place DataFrame storage.
            Fs: Sampling frequency in Hz. Defaults to ``self.Fs`` with a
                warning if not specified.

        Returns:
            Demodulated IQ signal or updated DataFrame.
        """
        if Fs is None:
            Fs = self.Fs
            if self.Warnings:
                print("Warning: (demodulate) No sampling frequency specified, using default Fs of {} MSps.".format(Fs / 1e6))
        return self.inputCheck(frame, method=self._demodulate, col_name=col_name, args={"Fs": Fs})

    def _removeDC(self, input):
        return input - np.average(input)

    def removeDC(self, frame: np.ndarray | pd.DataFrame = None, col_name=None):
        """Remove the DC offset from an IQ signal by subtracting its mean.

        Args:
            frame: Input IQ data as a NumPy array or DataFrame.
            col_name: Optional column name for in-place DataFrame storage.

        Returns:
            DC-free signal or updated DataFrame.
        """
        return self.inputCheck(frame, method=self._removeDC, col_name=col_name)

    #Github Copilot wrote this function, not sure if it works!
    def _findPeaks(self, input):
        return np.where(np.diff(np.sign(np.diff(input))))[0] + 1

    def findPeaks(self, frame: np.ndarray | pd.DataFrame = None, col_name=None):
        """Find local extrema (peaks and troughs) in a signal.

        Args:
            frame: Input signal as a NumPy array or DataFrame.
            col_name: Optional column name for in-place DataFrame storage.

        Returns:
            Array of indices at which the signal has local extrema, or an
            updated DataFrame.
        """
        return self.inputCheck(frame, method=self._findPeaks, col_name=col_name)

    def _reconstruct(self, input):
        cos = np.real(input) * np.sin(2 * np.pi * self.Fc * np.linspace(1, len(input), len(input)) / self.Fs)
        sin = np.imag(input) * np.cos(2 * np.pi * self.Fc * np.linspace(1, len(input), len(input)) / self.Fs)
        return cos + sin

    def reconstruct(self, frame: np.ndarray | pd.DataFrame = None, col_name=None):
        """Reconstruct the real passband signal from a complex baseband IQ frame.

        Args:
            frame: Complex IQ signal as a NumPy array or DataFrame.
            col_name: Optional column name for in-place DataFrame storage.

        Returns:
            Real-valued passband signal or updated DataFrame.
        """
        return self.inputCheck(frame, method=self._reconstruct, col_name=col_name)

    def _unwrapPhase(self, input):
        phase = np.unwrap(input)
        return phase

    def unwrapPhase(self, frame: np.ndarray | pd.DataFrame = None, col_name=None):
        """Compute the unwrapped phase of a signal.

        Args:
            frame: Input signal (typically the phase of an IQ signal) as a
                NumPy array or DataFrame.
            col_name: Optional column name for in-place DataFrame storage.

        Returns:
            Unwrapped phase array or updated DataFrame.
        """
        return self.inputCheck(frame, method=self._unwrapPhase, col_name=col_name)

    def _gradient(self, input):
        return np.gradient(input)

    def gradient(self, frame: np.ndarray | pd.DataFrame = None, col_name=None):
        """Compute the numerical gradient of a signal.

        Args:
            frame: Input signal as a NumPy array or DataFrame.
            col_name: Optional column name for in-place DataFrame storage.

        Returns:
            Gradient array or updated DataFrame.
        """
        return self.inputCheck(frame, method=self._gradient, col_name=col_name)

    def _sinc(self, input, length=30):
        t = np.linspace(.1, 1, length)
        lpf = np.sinc(t)
        return np.convolve(input, lpf)

    def sinc(self, frame: np.ndarray | pd.Series = None, col_name=None, length=None):
        """Apply a sinc (low-pass) FIR filter to a signal.

        Args:
            frame: Input signal as a NumPy array or Series.
            col_name: Optional column name for in-place DataFrame storage.
            length: Number of taps in the sinc filter. Defaults to 30 with a
                warning if not specified.

        Returns:
            Filtered signal or updated DataFrame.
        """
        if length is None:
            length = 30
            if self.Warnings:
                print("Warning: (sinc) No filter length specified, using default length of {}".format(length))
        return self.inputCheck(frame, method=self._sinc, col_name=col_name, args={"length": length})

    def _butter(self, input, cutoff=1e6, Fs=Fs):
        fltr = scipy.signal.butter(30, cutoff, 'low', analog=False, output='sos', fs=Fs)
        return scipy.signal.sosfilt(fltr, input)

    def butter(self, frame: np.ndarray | pd.Series = None, col_name=None, cutoff=None, Fs=None):
        """Apply a 30th-order Butterworth low-pass filter to a signal.

        Args:
            frame: Input signal as a NumPy array or Series.
            col_name: Optional column name for in-place DataFrame storage.
            cutoff: Filter cutoff frequency in Hz. Defaults to 1 MHz with a
                warning if not specified.
            Fs: Sampling frequency in Hz. Defaults to ``self.Fs`` with a
                warning if not specified.

        Returns:
            Filtered signal or updated DataFrame.
        """
        if cutoff is None:
            cutoff = 1e6
            if self.Warnings:
                print("Warning: (butter) No filter cutoff specified, using default cutoff of {}MHz".format(cutoff / 1e6))
        if Fs is None:
            Fs = self.Fs
            if self.Warnings:
                print("Warning: (butter) No filter sampling frequency specified, using default Fs of {}Msps".format(Fs / 1e6))
        return self.inputCheck(frame, method=self._butter, col_name=col_name, args={"cutoff": cutoff, "Fs": Fs})

    def _smooth(self, input, window_len=11,window='hanning'):
        """smooth the data using a window with requested size.
        This method is based on the convolution of a scaled window on the signal.
        The signal is prepared by introducing reflected copies of the signal 
        (with the window size) in both ends so that transient parts are minimized
        in the begining and end part of the output signal.
        input:
            window_len: the dimension of the smoothing window; 
                        should be an odd integer
            window: the type of window from 'flat', 'hanning', 'hamming', 
                    'bartlett', 'blackman'
                    flat window will produce a moving average smoothing.
        output:
            the smoother FIR filter

        see also: 
        numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, 
        numpy.convolve scipy.signal.lfilter"""
            
        # if x.ndim != 1:
        #     raise ValueError( "smooth only accepts 1 dimension arrays.")

        # if x.size < window_len:
        #     raise ValueError( "Input vector needs to be bigger than window size.")
        
        # if window_len<3:
        #     return x
        
        if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
            raise ValueError( f"Window is on of '{'flat', 'hanning', 'hamming', 'bartlett', 'blackman'}'")
        
        # s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
        
        if window == 'flat': #moving average
            w=np.ones(window_len,'d')
        else:
            w=eval( f"np.{window}(window_len)")
        
        lpf = w/w.sum()
        res = np.convolve(input,lpf)
        return res[int(len(lpf)/2-1):-int(len(lpf)/2)]
    
    def smooth(self, frame: np.ndarray | pd.Series = None, col_name=None, window_len=11, window='hanning'):
        """Smooth a signal using a convolution window.

        Args:
            frame: Input signal as a NumPy array or Series.
            col_name: Optional column name for in-place DataFrame storage.
            window_len: Length of the smoothing window (should be odd).
            window: Window type: ``'flat'``, ``'hanning'``, ``'hamming'``,
                ``'bartlett'``, or ``'blackman'``. A flat window produces a
                moving-average smoother.

        Returns:
            Smoothed signal or updated DataFrame.
        """
        return self.inputCheck(frame, method=self._smooth, col_name=col_name, args={"window_len": window_len, "window": window})

    def _downSample(self, input, downSampleRate=2, shift=0):
        return input[shift::downSampleRate]

    def downSample(self, frame: np.ndarray | pd.DataFrame = None, downSampleRate=2, shift=0, col_name=None):
        """Decimate a signal by keeping every *downSampleRate*-th sample.

        Args:
            frame: Input signal as a NumPy array or DataFrame.
            downSampleRate: Decimation factor (keeps every N-th sample).
            shift: Starting offset before decimation.
            col_name: Optional column name for in-place DataFrame storage.

        Returns:
            Decimated signal or updated DataFrame.
        """
        return self.inputCheck(frame, method=self._downSample, col_name=col_name, args={"downSampleRate": downSampleRate, "shift": shift})

    def _upSample(self, input, upSampleRate=2):
        return np.repeat(input, upSampleRate)

    def upSample(self, frame: np.ndarray | pd.DataFrame = None, upSampleRate=2, col_name=None):
        """Upsample a signal by repeating each sample *upSampleRate* times.

        Args:
            frame: Input signal as a NumPy array or DataFrame.
            upSampleRate: Interpolation factor.
            col_name: Optional column name for in-place DataFrame storage.

        Returns:
            Upsampled signal or updated DataFrame.
        """
        return self.inputCheck(frame, method=self._upSample, col_name=col_name, args={"upSampleRate": upSampleRate})

    def _scalePhaseGradientToHz(self, input, Fs=Fs):
        return input * Fs / (2 * np.pi)

    def scalePhaseGradientToHz(self, frame: np.ndarray | pd.DataFrame = None, col_name=None, Fs=None):
        """Convert a phase gradient signal from radians/sample to Hz.

        Scales the instantaneous frequency estimate (phase gradient) by
        ``Fs / (2π)`` so the output is in hertz.

        Args:
            frame: Phase gradient signal as a NumPy array or DataFrame.
            col_name: Optional column name for in-place DataFrame storage.
            Fs: Sampling frequency in Hz. Defaults to ``self.Fs`` with a
                warning if not specified.

        Returns:
            Frequency-deviation signal in Hz or updated DataFrame.
        """
        if Fs is None:
            Fs = self.Fs
            if self.Warnings:
                print("Important Warning: (scalePhaseGradientToHz) No sampling frequency specified, using default Fs of {}Msps.".format(Fs / 1e6))
        return self.inputCheck(frame, method=self._scalePhaseGradientToHz, col_name=col_name, args={"Fs": Fs})

    def keepPositive(self, samples):
        """Zero out all negative values in *samples*, keeping only positives.

        Args:
            samples: NumPy array of signal values.

        Returns:
            Copy of *samples* with negative values replaced by zero.
        """
        sam = samples.copy()
        sam[sam < 0] = 0
        return sam

    def keepNegative(self, samples):
        """Zero out all positive values in *samples*, keeping only negatives.

        Args:
            samples: NumPy array of signal values.

        Returns:
            Copy of *samples* with positive values replaced by zero.
        """
        sam = samples.copy()
        sam[sam > 0] = 0
        return sam


    def _bitMetaDataGenerator(self, sample , indx, bitsPerSample):
        return [
                {
                    'samples': sample[x[0]:x[1]],
                    'numberOfBits':round((x[1] - x[0])/bitsPerSample), 
                    'indxBegining': x[0], 'indxEnd': x[1], 
                    'len': x[1] - x[0], 'slope':1,
                    'overshoot': np.max(np.abs(sample[x[0]:x[1]])),
                    # 'undershoot': np.min(np.abs(sample[x[0]:x[1]])),
                    'std': np.std(sample[x[0]:x[1]]),
                    'mean': np.mean(np.abs(sample[x[0]:x[1]])),
                } 
                for x in indx if np.max(np.abs(sample[x[0]:x[1]])) > 60000 ## will remove the scaled phase gradient bits that are not bits  
               ]


    def nonZeroGrouper(self, samples, biggerThan=None, smallerThan=None, Fs=Fs, noGroupBefore=None):
        """Group consecutive non-zero sample runs within size bounds.

        Finds contiguous runs of non-zero values in *samples* and returns
        the ``[start, end]`` index pairs for runs whose length falls between
        *biggerThan* and *smallerThan* and starts after *noGroupBefore*.

        Args:
            samples: 1-D NumPy array (typically a zeroed phase-gradient
                signal after :meth:`keepPositive` or :meth:`keepNegative`).
            biggerThan: Minimum run length (samples). Defaults to
                ``10 * Fs / self.Fs``.
            smallerThan: Maximum run length (samples). Defaults to
                ``10000 * Fs / self.Fs``.
            Fs: Sampling frequency used to scale the thresholds.
            noGroupBefore: Sample index before which runs are discarded.
                Defaults to 0.

        Returns:
            NumPy array of shape ``(N, 2)`` containing ``[start, end]``
            index pairs for each qualifying run.
        """
        if smallerThan is None:
            smallerThan = 10000 * Fs / self.Fs
        if biggerThan is None:
            biggerThan = 10 * Fs / self.Fs
        if noGroupBefore is None:
            noGroupBefore = 0*Fs/self.Fs
            if self.Warnings:
                print("Warning: (nonZeroGrouper) No noGroupBefore specified, using default noGroupBefore of {}".format(noGroupBefore) )

        test_list = np.nonzero(samples)
        framesIndex = []
        for k, g in groupby(enumerate(test_list[0]), lambda ix: ix[0]-ix[1]):
            temp = list(map(itemgetter(1), g))
            if len(temp)< biggerThan or len(temp)> smallerThan:
                continue
            if temp[0] > noGroupBefore: # no bits befor 1100 samples at 100e6 sample rate
                framesIndex.append([temp[0],temp[-1]])
        return np.array(framesIndex)



    # the default values are for 100Msps
    # the BLE symbol rate is 1 million symbol per second. 
    def _bitFinderFromPhaseGradient(self, sample, Fs= Fs, bitsPerSample = None, biggerThan = None, smallerThan = None , noGroupBefore = None, plot = False, col_name = None, title = None, x_label = None, y_label = None):
        if bitsPerSample is None:
            bitsPerSample = Fs/1e6
            if self.Warnings:
                print("Warning: (bitFinderFromPhaseGradient) No bits per sample specified, using default bitsPerSample of {}".format(bitsPerSample) )
        if biggerThan is None:
            biggerThan = int(0.82*bitsPerSample)
            if self.Warnings:
                print("Warning: (bitFinderFromPhaseGradient) No frame bigger than specified, using default biggerThan of {}".format(biggerThan) )
        if smallerThan is None:
            smallerThan = int(100*bitsPerSample)
            if self.Warnings:
                print("Warning: (bitFinderFromPhaseGradient) No frame smaller than specified, using default smallerThan of {}".format(smallerThan) )
        
        X_positive = self.keepPositive(sample)
        X_negative = self.keepNegative(sample)
        pIndx = self.nonZeroGrouper(X_positive, Fs=Fs, biggerThan = biggerThan, smallerThan = smallerThan, noGroupBefore = noGroupBefore)
        nIndx = self.nonZeroGrouper(X_negative, Fs=Fs, biggerThan = biggerThan, smallerThan = smallerThan, noGroupBefore = noGroupBefore)
        pIndx_meta = self._bitMetaDataGenerator(sample, pIndx, bitsPerSample)
        nIndx_meta = self._bitMetaDataGenerator(sample, nIndx, bitsPerSample)
    
        if plot:
            plt.figure(figsize=(20,3), dpi=100)
            plt.plot(np.zeros(max(len(X_positive),len(X_negative))))
            plt.plot(sample)
            plt.stem(pIndx.flatten(), [.3*np.max(X_positive)]*len(pIndx.flatten()) ,'r')
            plt.stem(nIndx.flatten(), [.3*np.min(X_negative)]*len(nIndx.flatten()))
            plt.title(title)
            plt.xlabel(x_label if x_label is not None else 'Samples')
            plt.ylabel(y_label if y_label is not None else 'Freq. Deviation (Hz)')
            plt.show()
            plt.close()
        return pd.DataFrame(sorted(pIndx_meta + nIndx_meta, key=lambda x: x["indxBegining"]))
    
    def bitFinderFromPhaseGradient(self, frame: np.ndarray | pd.Series = None, col_name=None, Fs=None, bitsPerSample=None, biggerThan=None, smallerThan=None, noGroupBefore=None, plot=False, title=None, x_label=None, y_label=None):
        """Identify BLE bit boundaries from a scaled phase-gradient signal.

        Separates positive and negative frequency deviations (``+`` and ``-``
        BLE symbols), groups consecutive runs with :meth:`nonZeroGrouper`,
        and returns a sorted DataFrame of bit-segment metadata. Optionally
        plots the signal with annotated bit boundaries.

        Args:
            frame: Scaled phase-gradient signal (Hz) as a NumPy array or
                Series.
            col_name: Optional column name for in-place DataFrame storage.
            Fs: Sampling frequency in Hz. Defaults to ``self.Fs``.
            bitsPerSample: Expected number of samples per bit symbol.
                Defaults to ``Fs / 1e6`` (100 samples at 100 Msps).
            biggerThan: Minimum run length in samples.
            smallerThan: Maximum run length in samples.
            noGroupBefore: Ignore runs starting before this sample index.
            plot: If True, display a stem plot of detected bit boundaries.
            title: Plot title (used when *plot* is True).
            x_label: X-axis label (used when *plot* is True).
            y_label: Y-axis label (used when *plot* is True).

        Returns:
            ``pd.DataFrame`` where each row describes one detected bit
            segment with keys ``samples``, ``numberOfBits``,
            ``indxBegining``, ``indxEnd``, ``len``, ``slope``,
            ``overshoot``, ``std``, and ``mean``.
        """
        if Fs is None:
            Fs = self.Fs
            if self.Warnings:
                print("IMPORTANT WARNING: (bitFinderFromPhaseGradient) No sampling frequency specified, using default Fs of {}Msps.".format(Fs / 1e6))
        return self.inputCheck(frame, method=self._bitFinderFromPhaseGradient, col_name=col_name, args={"Fs": Fs, "bitsPerSample": bitsPerSample, "noGroupBefore": noGroupBefore, "biggerThan": biggerThan, "smallerThan": smallerThan, "plot": plot, "title": title, "x_label": x_label, "y_label": y_label})

    
    

    def _plotUtills(self, input, title = None, x_label = None, y_label = None,x = None, xscale = None):
        plt.figure(figsize=self.figsize,dpi=self.dpi)

        if x is not None:
            plt.plot(np.linspace(x[0],x[1], len(input)),input)
        else:
            plt.plot(input)

        if xscale is not None:
            plt.xscale('symlog', linthreshx=xscale)

        if title is not None:
            plt.title(title)
        if x_label is not None:
            plt.xlabel(x_label)
        if y_label is not None:
            plt.ylabel(y_label)
        plt.show()

    def _plot(self, input, title = None, x_label = None, y_label = None, x = None, xscale = None):
        if isinstance(input, pd.Series):
            for column in input:
                self._plotUtills(input=column, title = title, x_label = x_label, y_label = y_label, x = x,  xscale = xscale)
        else:
            self._plotUtills(input=input, title = title, x_label = x_label, y_label = y_label, x = x,  xscale = xscale)
        
    def plot(self, frame: np.ndarray | pd.Series | pd.DataFrame = None, col_name: str | list = None, title: str = None, x_label: str = None, y_label: str = None, x: np.ndarray = None, xscale=None):
        """Plot one or more IQ signals or derived quantities.

        Delegates to :meth:`_plotUtills` for NumPy arrays and iterates over
        columns when *frame* is a DataFrame.

        Args:
            frame: Signal data to plot. Accepts a NumPy array, Series, or
                DataFrame. Falls back to ``self.df`` if None.
            col_name: Column(s) of a DataFrame to plot.
            title: Plot title.
            x_label: X-axis label.
            y_label: Y-axis label.
            x: Optional ``[min, max]`` pair to use as the x-axis range
                instead of sample indices.
            xscale: Symmetric-log threshold for the x-axis scale (optional).
        """
        args = {'title': title, 'x_label': x_label, 'y_label': y_label, 'x': x, 'xscale': xscale}
        self.inputCheck(frame, method=self._plot, col_name=col_name, args=args, plot=True)


    # def _apply(self,  method, input = None,col_name = None,args = None):
    #     print(args)
    #     if args is not None:
    #         return method(input, col_name, **args)
    #     else:
    #         return method(input, col_name)

    
    def apply(self, methods: list | dict, frame: np.ndarray | pd.Series | pd.DataFrame = None, col_name: str | list = None):
        """Apply a sequence of IQ processing methods to *frame* in order.

        Accepts either a list or a dict of methods:

        - **list**: Methods are applied in reverse pop order (LIFO). Each
          entry may be a string (looked up on ``self``) or a callable.
        - **dict**: Keys are method names (strings or callables); values are
          dicts of keyword arguments (or None for no extra args).

        User-defined callables (those whose ``__qualname__`` does not start
        with ``IQ.``) are dispatched via :meth:`inputCheck` directly.
        Built-in IQ methods are called with ``frame`` and ``col_name`` as
        keyword arguments plus any extra args from the dict value.

        Args:
            methods: Ordered collection of processing steps. See description
                above for the list vs dict semantics.
            frame: Input IQ data. Falls back to ``self.df`` if None.
            col_name: Column name for in-place DataFrame storage.

        Returns:
            Transformed *frame* after all methods have been applied.
        """
        if isinstance(methods, dict):
            method_keys = list(methods.keys())
            while len(method_keys) > 0:
                method_nm = method_keys.pop()
                if isinstance(method_nm, str):
                    method = self.__getattribute__(method_nm)
                else:
                    method = method_nm
                try:
                    method.__qualname__.startswith('IQ.')
                except:
                    frame = self.inputCheck(frame, method=method, col_name = col_name, args = methods[method_nm])
                    continue
                if not method.__qualname__.startswith('IQ.'): #User defined function
                    frame = self.inputCheck(frame, method=method, col_name = col_name, args = methods[method_nm])
                    continue
            
                if methods[method_nm] is not None: # if args is not None
                    try:
                        frame = method(frame = frame, col_name = col_name, **methods[method_nm])
                    except:
                        if self.Warnings:
                            print("**** Warning: args not applied ****")
                        frame = method(frame = frame, col_name = col_name)
                else:
                    frame = method(frame = frame, col_name = col_name)
                
        elif isinstance(methods, list):
            while len(methods) > 0:
                method_nm = methods.pop()
                if isinstance(method_nm, str):
                    method = self.__getattribute__(method_nm)
                else:
                    method = method_nm

                try:
                    method.__qualname__.startswith('IQ.')
                except:
                    frame = self.inputCheck(frame, method=method, col_name = col_name)
                    continue
                if not method.__qualname__.startswith('IQ.'): #User defined function
                    frame = self.inputCheck(frame, method=method, col_name = col_name)
                    continue
                frame = method(frame = frame, col_name = col_name)

        return frame



    

        
         