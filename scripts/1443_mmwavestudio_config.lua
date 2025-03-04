-- RT3 installation path
RSTD_PATH = RSTD.GetRstdPath()

-- Declare the loading function
dofile(RSTD_PATH .. "\\Scripts\\AR1xFunctions.lua")


-- Options for DUT Version
DUT_VER = {AR1xxx = 1}
DUT_VERSION = DUT_VER.AR1xxx
-- Set automoation mode on/off (no message boxes)
local automation_mode = true
Load_AR1x_Client(automation_mode)

-- Radar Settings (Original)
-- 3 Tx 4 Rx | complex 1x
	
-------- SET THESE CONSTANTS PLEASE --------

COM_PORT = 9 
RADARSS_PATH = "C:\\ti\\mmwave_studio_02_01_01_00\\rf_eval_firmware\\radarss\\xwr12xx_xwr14xx_radarss_ES2.0.bin"
MASTERSS_PATH = "C:\\ti\\mmwave_studio_02_01_01_00\\rf_eval_firmware\\masterss\\xwr12xx_xwr14xx_masterss_ES2.0.bin"
SAVE_DATA_PATH = "C:\\Users\\oyvindmb\\radar_logs\\adc_data.bin"
DUMP_DATA_PATH = "C:\\Users\\oyvindmb\\radar_logs\\adc_data_RAW_0.bin"
PKT_LOG_PATH  = "C:\\Users\\oyvindmb\\radar_logs\\pktlogfile.txt"

--------------------------------------------

-------- VERY IMPORTANT AND SERIOUS RADAR SETTINGS --------
-- General
NUM_TX = 3
NUM_RX = 4

-- ProfileConfig
START_FREQ = 77 -- GHz
IDLE_TIME = 100 -- us
RAMP_END_TIME = 62 -- us
ADC_START_TIME = 6 --us
FREQ_SLOPE = 60.012 -- MHz/us
ADC_SAMPLES = 128
SAMPLE_RATE = 5000 -- ksps
RX_GAIN = 30 -- dB

-- ChirpConfig
-- the setup is such that we receive Rx information in the order of Tx1->Tx3->Tx2
-- this translates to getting all the azimuth information first (indices [0,7]) then getting any elevation information (indices [8,11])

-- FrameConfig
START_CHIRP_TX = 0
END_CHIRP_TX = 2 -- 2 for 3 tx
NUM_FRAMES = 0 -- Set this to 0 to continuously stream data
CHIRP_LOOPS = 64 --    
PERIODICITY = 70 -- ms  
-----------------------------------------------------------

-------- THIS IS FINE --------
ar1.FullReset()
ar1.SOPControl(2)
ar1.Connect(COM_PORT,921600,1000)
------------------------------

-------- https://cdn.vox-cdn.com/thumbor/2q97YCXcLOlkoR2jKKEMQ-wkG9k=/0x0:900x500/1200x800/filters:focal(378x178:522x322)/cdn.vox-cdn.com/uploads/chorus_image/image/49493993/this-is-fine.0.jpg --------
ar1.Calling_IsConnected()
ar1.SelectChipVersion("XWR1443")
ar1.frequencyBandSelection("77G")
------------------------------------------------------------------------------------------------------------------------------------------------------------------------

-------- DOWNLOAD FIRMARE --------
ar1.DownloadBSSFw(RADARSS_PATH)
ar1.GetBSSFwVersion()
ar1.GetBSSPatchFwVersion()
ar1.DownloadMSSFw(MASTERSS_PATH)
ar1.PowerOn(0, 1000, 0, 0)
ar1.RfEnable()
ar1.GetBSSFwVersion()
--------

-------- STATIC CONFIG STUFF --------
ar1.ChanNAdcConfig(1, 1, 1, 1, 1, 1, 1, 2, 1, 0)
ar1.LPModConfig(0, 0)
ar1.RfInit()
--------------------------------------

-------- DATA CONFIG STUFF --------
ar1.DataPathConfig(513, 1216644097, 0)
ar1.LvdsClkConfig(1, 1)
ar1.LVDSLaneConfig(0, 1, 1, 1, 1, 1, 0, 0)
-----------------------------------

-------- SENSOR CONFIG STUFF --------
ar1.ProfileConfig(0, START_FREQ, IDLE_TIME, ADC_START_TIME, RAMP_END_TIME, 0, 0, 0, 0, 0, 0, FREQ_SLOPE, 0, ADC_SAMPLES, SAMPLE_RATE, 0, 0, RX_GAIN)
ar1.ChirpConfig(0, 0, 0, 0, 0, 0, 0, 1, 0, 0)
ar1.ChirpConfig(1, 1, 0, 0, 0, 0, 0, 0, 0, 1)
ar1.ChirpConfig(2, 2, 0, 0, 0, 0, 0, 0, 1, 0)
ar1.FrameConfig(START_CHIRP_TX, END_CHIRP_TX, NUM_FRAMES, CHIRP_LOOPS, PERIODICITY, 0, 0, 1)
-------------------------------------

-------- ETHERNET STUFF --------
ar1.SelectCaptureDevice("DCA1000")
ar1.CaptureCardConfig_EthInit("192.168.33.30", "192.168.33.180", "12:34:56:78:90:12", 4096, 4098)
ar1.CaptureCardConfig_Mode(1, 1, 1, 2, 3, 30)
ar1.CaptureCardConfig_PacketDelay(25)
--------------------------------



--ar1.CaptureCardConfig_StartRecord(SAVE_DATA_PATH, 1)
ar1.StartFrame()

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 

-------- CALCULATED AND NOT TOO SERIOUS PARAMETERS --------
CHIRPS_PER_FRAME = (END_CHIRP_TX - START_CHIRP_TX + 1) * CHIRP_LOOPS
NUM_DOPPLER_BINS = CHIRPS_PER_FRAME / NUM_TX
NUM_RANGE_BINS = ADC_SAMPLES
RANGE_RESOLUTION = (3e8 * SAMPLE_RATE * 1e3) / (2 * FREQ_SLOPE * 1e12 * ADC_SAMPLES)
MAX_RANGE = (300 * 0.9 * SAMPLE_RATE) / (2 * FREQ_SLOPE * 1e3)
DOPPLER_RESOLUTION = 3e8 / (2 * START_FREQ * 1e9 * (IDLE_TIME + RAMP_END_TIME) * 1e-6 * NUM_DOPPLER_BINS * NUM_TX)
MAX_DOPPLER = 3e8 / (4 * START_FREQ * 1e9 * (IDLE_TIME + RAMP_END_TIME) * 1e-6 * NUM_TX)


print("Chirps Per Frame:", CHIRPS_PER_FRAME)
print("Num Doppler Bins:", NUM_DOPPLER_BINS)
print("Num Range Bins:", NUM_RANGE_BINS)
print("Range Resolution:", RANGE_RESOLUTION)
print("Max Unambiguous Range:", MAX_RANGE)
print("Doppler Resolution:", DOPPLER_RESOLUTION)
print("Max Doppler:", MAX_DOPPLER)