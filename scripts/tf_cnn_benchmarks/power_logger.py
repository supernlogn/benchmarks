""" Descr:
      A power consumption logger for nvidia gpus in a system using python.
    author: 
      Ioannis Athanasiadis (supernlogn)
"""

import numpy as np
import subprocess
import threading
import time, datetime
import re
from json import dump

class NvidiaPowerReader(object):
  def __init__(self, time_step=1, num_gpus=1):
    self.p = subprocess.Popen("nvidia-smi -l " + str(time_step), shell=True, stdout=subprocess.PIPE)  
    self.pattern = re.compile("[0-9]*?[W]{1,1}[\s]{1,1}[/]{1,1}[\s]{1,1}[0-9]*?[W]{1,1}")
    self.measurements = []
    self.time_step = time_step
    self.num_gpus = num_gpus
    self.last_measurements = []
  def read_once(self):
    """
    read one measurement from all available gpus
    """
    time_start = time.time()
    i = 0
    s = "123123123123123"
    measurement_record = []
    while( len(s) >= 10 and time.time() - time_start < self.time_step):
      s = self.p.stdout.readline()
      m = re.findall(self.pattern, s)
      if( len(m) != 0 ):
        measurement = float(m[0].partition('W /')[0])
        if( i == 0 ):
          measurement_record.append(datetime.datetime.now())
          measurement_record.append(measurement)
        else:
          measurement_record.append(measurement)
        i += 1
        if i == self.num_gpus:
          break
    while( len(measurement_record) != self.num_gpus + 1 ):
      measurement_record.append(-1.0)
    return measurement_record

  def read_multi_synch(self, num_reads):
    measurements = [self.read_once() for _ in range(num_reads)]
    self.last_measurements = measurements
    return measurements

  def read_multi_asynch(self, num_reads, delay=5):
    self.t = threading.Thread(target=self.read_multi_synch, args=(num_reads,))
    timer = threading.Timer(delay, self.t.start)
    timer.start()
    # results will be in self.last_measurements
    return
  
  def filter_measurements(self, measurements):
    filtered_measurements = []
    for m in measurements:
      if(not np.any(m[1:] == -1.0)):
        filtered_measurements.append(m)

    return np.array(filtered_measurements)
  
  def power_stats(self, measurements=[], filter=True):
    if(measurements == []):
      measurements = self.last_measurements      
    if filter:
      filtered_measurements = self.filter_measurements(measurements) 
    else:
      filtered_measurements = np.array(measurements)
    means = [np.mean(filtered_measurements[:,i]) for i in range(1, self.num_gpus+1)]
    vars = [np.std(filtered_measurements[:,i]) for i in range(1, self.num_gpus+1)]
    return means, vars

  def write_results_to_file(self, logname, measurements=[], filter=True):
    if(measurements == []):
      measurements = self.last_measurements      
    if filter:
      filtered_measurements = self.filter_measurements(measurements) 
    else:
      filtered_measurements = np.array(measurements)

    with open(logname + "_power_log.json", "w") as f:
      obj_to_dump = {"times": [str(timestamp) for timestamp in filtered_measurements[:,0].tolist()]}
      for i in range(1, self.num_gpus+1):
        obj_to_dump["gpu_%d" % i] = filtered_measurements[:,i].tolist()
      means, vars = self.power_stats(filtered_measurements, filter=False)
      obj_to_dump["mean_powers"] = means
      obj_to_dump["vars"] = vars    
      dump(obj_to_dump, f)

  def stop(self):
    self.t.join()
    self.p.kill()
