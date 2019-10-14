import subprocess
import time
import traceback

sleep_time = 3600
url = "http://178.239.225.102/axis-cgi/mjpg/video.cgi"

i = 0
while True:
	try:
		proc = subprocess.Popen(["ffmpeg", "-i", url, "output_"+str(i)+".mp4"], stdin=subprocess.PIPE,
		                            stdout=subprocess.PIPE,
		                            stderr=subprocess.PIPE)
		time.sleep(sleep_time)
		print ("done waiting")
		proc.communicate(b'q\n')
		proc.wait()
		print ("done")
	except:
		traceback.print_exc()
		continue
	i+=1