# Copyright 2019 The OpenRadar Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import mmwave.dsp as dsp
import mmwave.clustering as clu
from mmwave.dataloader import DCA1000
import time
import matplotlib.pyplot as plt
import os
import subprocess

plt.close('all')

# QOL settings
RECORD_START_CMD_CODE = '0500'
numADCSamples = 128
numTxAntennas = 3
numRxAntennas = 4
numLoopsPerFrame = 64
numChirpsPerFrame = numTxAntennas * numLoopsPerFrame
numRangeBins = numADCSamples
numDopplerBins = numLoopsPerFrame
numAngleBins = 64

range_resolution, bandwidth = dsp.range_resolution(numADCSamples)
doppler_resolution = dsp.doppler_resolution(bandwidth)

plotRangeDopp = False
plot2DscatterXY = True
plot2DscatterXZ = False
plotCustomPlt = False

visTrigger = plot2DscatterXY + plot2DscatterXZ + plotRangeDopp + plotCustomPlt
assert visTrigger < 2, "Can only choose to plot one type of plot at once"

num_packets = 0
start_time = time.time()

if __name__ == '__main__':
    process = subprocess.Popen(r'C:\ti\mmwave_studio_02_01_01_00\mmWaveStudio\RunTime\Run_mmWaveStudio.cmd', 
                 cwd=r'C:\ti\mmwave_studio_02_01_01_00\mmWaveStudio\RunTime')


    #while process.poll() == None:
        #Wait here 
    #    time.sleep(1)

    time.sleep(45)

    #Connect to the socket in the constructor
    dca = DCA1000()
    dca._send_command(RECORD_START_CMD_CODE)

    # (1.5) Required Plot Declarations
    if plot2DscatterXY or plot2DscatterXZ:
        fig = plt.figure()
    elif plotRangeDopp:
        fig = plt.figure()
    elif plotCustomPlt:
        print("Using Custom Plotting")

    
    while True:
        # (1) Reading in adc data
        adc_data = dca.read(timeout=5)
        frame = dca.organize(adc_data, num_chirps=numChirpsPerFrame, num_rx=numRxAntennas, num_samples=numADCSamples)
        num_packets+=1
        # (2) Range Processing
        from mmwave.dsp.utils import Window
        radar_cube = dsp.range_processing(frame, window_type_1d=Window.BLACKMAN)
        assert radar_cube.shape == (
        numChirpsPerFrame, numRxAntennas, numADCSamples), "[ERROR] Radar cube is not the correct shape!"

        # (3) Doppler Processing 
        det_matrix, aoa_input = dsp.doppler_processing(radar_cube, num_tx_antennas=numTxAntennas, clutter_removal_enabled=True)

        # --- Show output
        if plotRangeDopp:
            det_matrix_vis = np.fft.fftshift(det_matrix, axes=1)
            plt.imshow(det_matrix_vis / det_matrix_vis.max())
            plt.pause(0.05)
            plt.clf()

        # (4) Object Detection
        # --- CFAR, SNR is calculated as well.
        fft2d_sum = det_matrix.astype(np.int64)
        thresholdDoppler, noiseFloorDoppler = np.apply_along_axis(func1d=dsp.ca_,
                                                                axis=0,
                                                                arr=fft2d_sum.T,
                                                                l_bound=1.5,
                                                                guard_len=4,
                                                                noise_len=16)

        thresholdRange, noiseFloorRange = np.apply_along_axis(func1d=dsp.ca_,
                                                            axis=0,
                                                            arr=fft2d_sum,
                                                            l_bound=2.5,
                                                            guard_len=4,
                                                            noise_len=16)

        thresholdDoppler, noiseFloorDoppler = thresholdDoppler.T, noiseFloorDoppler.T
        det_doppler_mask = (det_matrix > thresholdDoppler)
        det_range_mask = (det_matrix > thresholdRange)

        # Get indices of detected peaks
        full_mask = (det_doppler_mask & det_range_mask)
        det_peaks_indices = np.argwhere(full_mask == True)

        # peakVals and SNR calculation
        peakVals = fft2d_sum[det_peaks_indices[:, 0], det_peaks_indices[:, 1]]
        snr = peakVals - noiseFloorRange[det_peaks_indices[:, 0], det_peaks_indices[:, 1]]

        dtype_location = '(' + str(numTxAntennas) + ',)<f4'
        dtype_detObj2D = np.dtype({'names': ['rangeIdx', 'dopplerIdx', 'peakVal', 'location', 'SNR'],
                                'formats': ['<i4', '<i4', '<f4', dtype_location, '<f4']})
        detObj2DRaw = np.zeros((det_peaks_indices.shape[0],), dtype=dtype_detObj2D)
        detObj2DRaw['rangeIdx'] = det_peaks_indices[:, 0].squeeze()
        detObj2DRaw['dopplerIdx'] = det_peaks_indices[:, 1].squeeze()
        detObj2DRaw['peakVal'] = peakVals.flatten()
        detObj2DRaw['SNR'] = snr.flatten()

        # Further peak pruning. This increases the point cloud density but helps avoid having too many detections around one object.
        detObj2DRaw = dsp.prune_to_peaks(detObj2DRaw, det_matrix, numDopplerBins, reserve_neighbor=True)

        # --- Peak Grouping
        detObj2D = dsp.peak_grouping_along_doppler(detObj2DRaw, det_matrix, numDopplerBins)
        SNRThresholds2 = np.array([[2, 23], [10, 11.5], [35, 16.0]])
        peakValThresholds2 = np.array([[4, 275], [1, 400], [500, 0]])
        detObj2D = dsp.range_based_pruning(detObj2D, SNRThresholds2, peakValThresholds2, numRangeBins, 0.5, range_resolution)

        

        azimuthInput = aoa_input[detObj2D['rangeIdx'], :, detObj2D['dopplerIdx']]

        x, y, z = dsp.naive_xyz(azimuthInput.T, num_tx=numTxAntennas)
        xyzVecN = np.zeros((numTxAntennas, x.shape[0]))
        xyzVecN[0] = x * range_resolution * detObj2D['rangeIdx']
        xyzVecN[1] = y * range_resolution * detObj2D['rangeIdx']
        xyzVecN[2] = z * range_resolution * detObj2D['rangeIdx']

        # (6) Visualization
        if plot2DscatterXY:
            #xyzVecN = xyzVecN[:, (np.abs(xyzVecN[2]) < 1.5)]

            plt.ylim(bottom=0, top=10)
            plt.ylabel('Range')
            plt.xlim(left=-4, right=4)
            plt.xlabel('Azimuth')
            plt.grid(visible=True)
            
            
            plt.scatter(xyzVecN[0], xyzVecN[1], c='b', marker='o', s=3)
            print(xyzVecN.shape)
            print("TOTAL ELAPSED TIME IS")
            print((time.time() - start_time))
            print("NUMBER OF PACKETS")
            print(num_packets)
            plt.pause(0.1)
            fig.clear() 