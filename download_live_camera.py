import subprocess
import time
import traceback
import os
from subprocess import Popen, PIPE, STDOUT
import signal

sleep_time = 3600
url = "http://178.239.225.102/axis-cgi/mjpg/video.cgi"

i = 0
while True:
	try:
		proc = subprocess.Popen(' '.join(["ffmpeg", "-i", url, "-t", "01:00:00.000" , "output_"+str(i)+".mp4"]), shell=True, stdin=PIPE, stdout=PIPE)
		time.sleep(sleep_time)
		for j in range(4):
			try:
				proc.communicate(input=b'q')
			except:
				traceback.print_exc()
				pass
			finally:
				time.sleep(1)
		if proc.poll() is None:
			os.kill(proc.pid, signal.SIGTERM)
			print ("Done with one video")
	except:
		traceback.print_exc()
		continue
	i+=1