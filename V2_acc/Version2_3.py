import numpy as np
import time
import platform
import threading
import lib.device_model as deviceModel
from lib.data_processor.roles.jy901s_dataProcessor import JY901SDataProcessor
from lib.protocol_resolver.roles.wit_protocol_resolver import WitProtocolResolver
import serial
import pickle 
from sklearn.mixture import GaussianMixture
from procesamiento_aceleracion import build_melspectrogram_from_acc
import queue

############HYPERPARAMETROS###################
SR=10 #Hz
N_FFT=16
HOP_LENGTH= int(N_FFT/2)
N_MELS = 10
###############################################


def setConfig(device):
    device.unlock()
    time.sleep(0.1)
    device.writeReg(0x03, 6)
    time.sleep(0.1)
    device.writeReg(0x23, 0)
    time.sleep(0.1)
    device.writeReg(0x24, 0)
    time.sleep(0.1)
    device.save()

def AccelerationCalibration(device):
    device.AccelerationCalibration()
    print("Acceleration Calibration made, press input to continue")

def onUpdate(deviceModel):
    return [deviceModel.getDeviceData("accX"), deviceModel.getDeviceData("accY"), deviceModel.getDeviceData("accZ")]

q = queue.Queue()

def get_queue():
  return q


class LogMelFeatureExtractor(object):
  """Provide uint8 log mel spectrogram slices from an AccRecorder object.

  This class provides one public method, get_next_spectrogram(), which gets
  a specified number of spectral slices from an AudioRecorder.
  """

  def __init__(self, num_frames_hop=1):
    self.N_FFT = N_FFT
    self.N_MELS = N_MELS
    self.HOP_LENGTH = HOP_LENGTH
    self.sample_rate_hz=SR
    self.spectrogram_window_length_seconds = self.N_FFT/SR
    self.spectrogram_hop_length_seconds = self.HOP_LENGTH/SR
    self.num_mel_bins = self.N_MELS
    self.frame_length_spectra = 1
    if self.frame_length_spectra % num_frames_hop:
        raise ValueError('Invalid num_frames_hop value (%d), '
                         'must devide %d' % (num_frames_hop,
                                             self.frame_length_spectra))
    self.frame_hop_spectra = num_frames_hop
    self._norm_factor = 3
    self._clear_buffers()

  def _clear_buffers(self):
    self._audio_buffer = np.array([]).reshape(0,1)
    self._spectrogram = np.zeros((self.frame_length_spectra, self.num_mel_bins),
                                 dtype=np.float32)

  def _spectrogram_underlap_samples(self, audio_sample_rate_hz):
    return int((self.spectrogram_window_length_seconds -
                self.spectrogram_hop_length_seconds) * audio_sample_rate_hz)

  def _frame_duration_seconds(self, num_spectra):
    return (self.spectrogram_window_length_seconds +
            (num_spectra - 1) * self.spectrogram_hop_length_seconds)

  def _compute_spectrogram(self, samples, sample_rate):
    """Compute log-mel spectrogram and dont scale it to uint8."""
    samples = samples.flatten()
    spectrogram = build_melspectrogram_from_acc(audio_ndarray=samples, sample_rate= sample_rate, n_fft=self.N_FFT, hop_length=int(self.HOP_LENGTH), n_mels=self.N_MELS).T
    spectrogram = spectrogram.numpy()
    #Pq nos da (3,10) y en principio la primera tupla es la mas proxima 
    return spectrogram[0] #En principio deberia ser un array de (1,10) con los spectograms de un frame 

  def _get_next_spectra(self, acc_recorder, num_spectra):
    """Returns the next spectrogram.

    Compute num_spectra spectrogram samples from an accDevice.
    Blocks until num_spectra spectrogram slices are available.

    Args:
      recorder: an device object from which to get raw acc samples.
      num_spectra: the number of spectrogram slices to return.

    Returns:
      num_spectra spectrogram slices computed from the samples.
    """
    required_audio_duration_seconds = self._frame_duration_seconds(num_spectra)
    required_num_samples = int(
        np.ceil(required_audio_duration_seconds *
                self.sample_rate_hz))
    audio_samples = np.concatenate(
        (self._audio_buffer,
         acc_recorder.get_readings(required_num_samples - len(self._audio_buffer))[0]))
    self._audio_buffer = audio_samples[
        required_num_samples -
        self._spectrogram_underlap_samples(self.sample_rate_hz):]
    spectrogram = self._compute_spectrogram(
        audio_samples[:required_num_samples], self.sample_rate_hz)
    #assert len(spectrogram) == num_spectra
    return spectrogram

  def get_next_spectrogram(self, recorder):
    """Get the most recent spectrogram frame.

    Blocks until the frame is available.

    Args:
      recorder: an AudioRecorder instance which provides the audio samples.

    Returns:
      The next spectrogram frame as a uint8 numpy array.
    """
    #vamos a probar de esta otra forma sin sliding window, ya que nosotros necesitamso sacar los melcepstrums y analizarlos de frame en frame
    self._spectrogram = self._get_next_spectra(recorder, self.frame_hop_spectra)
    
    # Return a copy of the internal state that's safe to persist and won't
    # change the next time we call this function.
    spectrogram = self._spectrogram.copy()
    return spectrogram
  
class AccelerometerRecorder:
    def __init__(self, device, buffer_size=100):
        """
        Initialize the AccelerometerRecorder.
        
        Parameters:
        - device: The accelerometer device object.
        - buffer_size: Maximum number of samples to store in the buffer.
        """
        self.device = device
        self.buffer_size = buffer_size
        self.buffer = []  # Circular buffer to store readings
        self.lock = threading.Lock()  # To synchronize access to the buffer
        self.stop_event = threading.Event()  # To stop the thread
        self.thread = threading.Thread(target=self._record_loop)
    
    def start(self):
        """Start the recording thread."""
        if not self.thread.is_alive():
            self.stop_event.clear()
            self.thread.start()
    
    def stop(self):
        """Stop the recording thread."""
        self.stop_event.set()
        self.thread.join()
    
    def _record_loop(self):
        """Continuously read data from the device and store it in the buffer."""
        while not self.stop_event.is_set():
            try:
                # Fetch new readings from the device
                new_data = self._read_from_device()
                # Update the buffer with thread-safe access
                with self.lock:
                    self.buffer.append(new_data)
                    if len(self.buffer) > self.buffer_size:
                        self.buffer = self.buffer[-self.buffer_size:]  # Keep only the latest buffer_size samples
            except Exception as e:
                print(f"Error in record loop: {e}")
            time.sleep(0.1)  # Adjust sampling rate as needed

    def _read_from_device(self):
        """Fetch readings from the accelerometer device."""
        # Example reading: replace this with your device's reading logic
        data = onUpdate(self.device)  #`onUpdate` gets one sample as [accX, accY, accZ]
        acc_x = data[:, 0]
        acc_y = data[:, 1]
        acc_z = data[:, 2]
        acc_magnitude = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
        return acc_magnitude 
    
    def get_readings(self, n_samples):
        """
        Retrieve the last n_samples from the buffer.
        
        Parameters:
        - n_samples: Number of samples to retrieve.
        
        Returns:
        - A numpy array of the last n_samples readings.
        """
        with self.lock:
            if len(self.buffer) < n_samples:
                raise ValueError(f"Not enough samples in buffer. Requested {n_samples}, but only {len(self.buffer)} available.")
            return np.array(self.buffer[-n_samples:])
   


    
    
    


###### MODELO GMM########
data = np.genfromtxt('/home/alvaro/TFG/V2_acc/medidas/normal3.txt', delimiter='\t', skip_header=1)

# en data la primera columna es nan en su totalidad
raw_data = data[:, 1:] #limpiamos data 
acc_x = raw_data[:, 0]
acc_y = raw_data[:, 1]
acc_z = raw_data[:, 2]
acc_magnitude = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
acc_magnitude -= np.mean(acc_magnitude)
melspectrogram = build_melspectrogram_from_acc(acc_magnitude=acc_magnitude, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS).T
gm = GaussianMixture(n_components= N_MELS).fit(melspectrogram) #se crea el modelo 
"fit(X) donde X es un array con shape (n_samples, n_features)"
threshold=-35

#########################



anomalia = False
deviceIsClosed = True
stop_event = threading.Event()
deviceIsOpened = False

while True:
    print("iniciar programa")
    if deviceIsClosed:
        device = deviceModel.DeviceModel(
            "JY901",
            WitProtocolResolver(),
            JY901SDataProcessor(),
            "51_0"
        )
        if platform.system().lower() == 'linux':
            device.serialConfig.portName = "/dev/ttyUSB0"
        else:
            device.serialConfig.portName = "COM39"
        device.serialConfig.baud = 9600

        try:
            device.openDevice()
            AccelerationCalibration(device)
            input()
            deviceIsClosed = False
            deviceIsOpened = True
        except serial.SerialException as e:
            print(f"Error opening serial port: {e}")
            continue  # Retry opening the device in the next iteration
    if deviceIsOpened:
        recorder = AccelerometerRecorder(device, buffer_size=100)
        try:
            # Start recording data
            recorder.start()
            # Wait for buffer to fill up
            time.sleep(1)  # Adjust based on your device's sampling rate
            # Fetch the last 10 samples
            readings = recorder.get_readings(10)
            print("Last 10 samples:", readings)
            deviceIsOpened= False
        except Exception as e:
            print(f"Error: {e}")

    try:
        featureExtractor = LogMelFeatureExtractor(num_frames_hop=1)
        timed_out = False
        while not timed_out:
            spectrogram = featureExtractor.get_next_spectrogram(recorder=recorder)
            p = gm.score_samples(spectrogram)
            if p<threshold:
                print("ANOMALY!!!!, score:{}".format(p))
            else:
                print("normal, score:{}".format(p))
    
    except serial.SerialException as e:
        print(f"Serial exception during read: {e}")
        if device.serialPort and device.serialPort.is_open:
            device.closeDevice()
        deviceIsClosed = True
    except Exception as e:
        print(f"Unexpected exception: {e}")
        if device.serialPort and device.serialPort.is_open:
            device.closeDevice()
        deviceIsClosed = True

    #time.sleep(0.5)
