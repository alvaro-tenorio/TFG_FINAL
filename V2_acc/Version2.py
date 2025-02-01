import numpy as np
from scipy.stats import multivariate_normal

#ACCELEROMETER
import time
import datetime
import platform
import struct
import threading
import lib.device_model as deviceModel
from lib.data_processor.roles.jy901s_dataProcessor import JY901SDataProcessor
from lib.protocol_resolver.roles.wit_protocol_resolver import WitProtocolResolver

terminate_event = threading.Event()

def setConfig(device):
    """
    Example setting configuration information
    :param device: Device model
    :return:
    """
    device.unlock()                # unlock
    time.sleep(0.1)                # Sleep 100ms
    device.writeReg(0x03, 6)       # Set the transmission back rate to 10HZ
    time.sleep(0.1)                # Sleep 100ms
    device.writeReg(0x23, 0)       # Set the installation direction: horizontal and vertical
    time.sleep(0.1)                # Sleep 100ms
    device.writeReg(0x24, 0)       # Set the installation direction: nine axis, six axis
    time.sleep(0.1)                # Sleep 100ms
    device.save()                  # Save

def AccelerationCalibration(device):
    """
    Acceleration calibration
    :param device:  Device model
    :return:
    """
    device.AccelerationCalibration()                 # Acceleration calibration
    print("Acceleration Calibration")

def onUpdate(deviceModel):
    """
    Data update event
    :param deviceModel:  Device model
    :return:
    """
    return [deviceModel.getDeviceData("accX"), deviceModel.getDeviceData("accY"),deviceModel.getDeviceData("accZ"),deviceModel.getDeviceData("gyroX"), deviceModel.getDeviceData("gyroY"), deviceModel.getDeviceData("gyroZ")]


def LoopReadThead(device):
    """
    循环读取数据  Cyclic read data
    :param device:
    :return:
    """
    while not terminate_event.is_set():                            #循环读取数据 Cyclic read data
        device.readReg(0x30, 41)            #读取 数据  Read data      

# MODELO ESTADISTICO
#funciones 
def multivariate_gaussian(dataset,mu,sigma):
    p = multivariate_normal(mean=mu, cov=sigma)
    return p.logpdf(dataset)   #log

def estimate_gaussian(dataset):
    mu =np.mean(dataset, axis =0)
    sigma = np.cov(dataset.T)

    return mu, sigma

data = np.genfromtxt('/home/alvaro/TFG/V2/ventilador_normal.txt', delimiter='\t', skip_header=1)
# en data la primera columna es nan en su totalidad
clean_data = data[:, 1:] #limpiamos data 
mu, sigma = estimate_gaussian(clean_data)
ep = -10 #umbral de anomal'ia

#IMPORTS
import os
import glob
import time

# FUNCIONES ACELEROMETRO

def read_accelerometer(device):
    t = threading.Thread(target=LoopReadThead, args=(device,))  #开启一个线程读取数据 Start a thread to read data
    t.start()
    time.sleep(1)  # Ensure the thread has started and device is reading
    readings = onUpdate(device)
    return readings

anomalia=False
deviceIsClosed = True
while  True:
    """
    Initialize a device model
    """
    if(deviceIsClosed):
        device = deviceModel.DeviceModel(
        "JY901",
        WitProtocolResolver(),
        JY901SDataProcessor(),
        "51_0"
        )
        if (platform.system().lower() == 'linux'):
            device.serialConfig.portName = "/dev/ttyUSB0"   #Set serial port
        else:
            device.serialConfig.portName = "COM39"          #Set serial port
        device.serialConfig.baud = 9600                     #Set baud rate
        device.openDevice()
        #AccelerationCalibration(device)
        deviceIsClosed = False
        
 
    
    acc_readings= read_accelerometer(device)
    p = multivariate_gaussian(acc_readings, mu, sigma)
    print(acc_readings, p)
    time.sleep(0.5)
    #capture = [acc_readings, p]
    #recordings.append(capture)
    #time.sleep(0.5)
    if p <= ep:
        print("ANOMALIAAAA")
        anomalia = True

    elif p > ep:
        print("no anomalia")







    


        

