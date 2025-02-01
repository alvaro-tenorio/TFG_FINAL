# coding:UTF-8
"""
"""
import time
import datetime
import platform
import struct
import threading
import lib.device_model as deviceModel
from lib.data_processor.roles.jy901s_dataProcessor import JY901SDataProcessor
from lib.protocol_resolver.roles.wit_protocol_resolver import WitProtocolResolver

_writeF = None                    # Write file
_IsWriteF = False                 # Write file identification
def readConfig(device):
    """
    Example of reading configuration information
    :param device: Device model
    :return:
    """
    tVals = device.readReg(0x02,3)  #读取数据内容、回传速率、通讯速率   Read data content, return rate, communication rate
    if (len(tVals)>0):
        print("tvals" + str(tVals))
    else:
        print("tvals not")
    tVals = device.readReg(0x23,2)  #读取安装方向、算法  Read the installation direction and algorithm
    if (len(tVals)>0):
        print("len tVals" + str(tVals))
    else:
        print("len tVals not")

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

    print("device calibrated")

def FiledCalibration(device):
    """
    Magnetic field calibration
    :param device:  Device model
    :return:
    """
    device.BeginFiledCalibration()                   # Starting field calibration
    if input("begin? (Y/N)").lower()=="y":
        device.EndFiledCalibration()                 # End field calibration
        print("end calibration")

def onUpdate(deviceModel):
    """
    Data update event
    :param deviceModel:  Device model
    :return:
    """
    """print("time:" + str(deviceModel.getDeviceData("Chiptime"))
         , " temp:" + str(deviceModel.getDeviceData("temperature"))
         , " acc" + str(deviceModel.getDeviceData("accX")) +","+  str(deviceModel.getDeviceData("accY")) +","+ str(deviceModel.getDeviceData("accZ"))
         , " gyro:" + str(deviceModel.getDeviceData("gyroX")) +","+ str(deviceModel.getDeviceData("gyroY")) +","+ str(deviceModel.getDeviceData("gyroZ"))
         )
    """
    data = "\t" + str(deviceModel.getDeviceData("accX")) + "\t" + str(deviceModel.getDeviceData("accY")) +"\t"+ str(deviceModel.getDeviceData("accZ"))+"\t" + str(deviceModel.getDeviceData("gyroX")) +"\t"+ str(deviceModel.getDeviceData("gyroY")) +"\t"+ str(deviceModel.getDeviceData("gyroZ"))+ "\n"
    if (_IsWriteF):

        _writeF.write(data)
         
def LoopReadThead(device):
    """
    循环读取数据  Cyclic read data
    :param device:
    :return:
    """
    while(True):                            #循环读取数据 Cyclic read data
        device.readReg(0x30, 41)            #读取 数据  Read data


def startRecord():
    """
    Start recording data
    :return:
    """
    global _writeF
    global _IsWriteF
    _writeF = open(str(datetime.datetime.now().strftime('%Y%m%d%H%M%S')) + ".txt", "w")    #新建一个文件
    _IsWriteF = True                                                                        #标记写入标识
    Tempstr = "Chiptime"
    Tempstr +=  "\tax(g)\tay(g)\taz(g)"
    Tempstr += "\twx(deg/s)\twy(deg/s)\twz(deg/s)"
    Tempstr += "\tAngleX(deg)\tAngleY(deg)\tAngleZ(deg)"
    Tempstr += "\tT()"
    Tempstr += "\tmagx\tmagy\tmagz"
    Tempstr += "\tlon\tlat"
    Tempstr += "\tYaw\tSpeed"
    Tempstr += "\tq1\tq2\tq3\tq4"
    Tempstr += "\r\n"
    _writeF.write(Tempstr)
    print("OLE start")

def endRecord():
    """
    End record data
    :return:
    """
    global _writeF
    global _IsWriteF
    _IsWriteF = False             # 标记不可写入标识    Tag cannot write the identity
    _writeF.close()               #关闭文件 Close file
    print("OLE end")

if __name__ == '__main__':

    print("welcome")
    """
    Initialize a device model
    """
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
    device.openDevice()                                 #Open serial port
    AccelerationCalibration(device)
    input()
    readConfig(device)                                  #Read configuration information
    startRecord()                                            # 开始记录数据   Start recording data
    t = threading.Thread(target=LoopReadThead, args=(device,))  #开启一个线程读取数据 Start a thread to read data
    t.start()
    device.dataProcessor.onVarChanged.append(onUpdate)  #Data update event
    input()
    device.closeDevice()
    endRecord()
