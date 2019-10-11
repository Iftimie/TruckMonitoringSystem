# TruckMonitoringSystem
> Work in progress

## API

## DataFlow

The way I designed this application and package is by working only with movies. The movie represents the initial source 
of data processing. Given that I love generators and decorators, I decided to define the data pipelines with them.

Use cases:

* On a new online request or on an offline processing, the app works only with one movie
* We may decide that not all frames needs to be processed, since the computation takes some time. We may decide that
processing one frame every other 5 frames is sufficient.
* The results of the processing are stored into a .csv file in the following way.	
```
	img_id	label	obj_id	score	        x1	x2	y1	y2
0	0							
1	1							
2	2							
3	3	truck	0	0.801649988	10	301	297	480
4	3	truck	1	0.667901516	266	378	270	372
5	3	car	2	0.957448542	53	96	118	141
6	3	car	3	0.89286685	378	439	126	165
7	3	car	4	0.886776984	204	262	97	125
8	3	car	5	0.872502923	17	77	153	181
9	3	car	6	0.751685441	159	186	136	163
10	4	truck	0	0.801649988	10	301	297	480
11	4	truck	1	0.667901516	266	378	269	372
12	4	car	2	0.957448542	48	89	118	141
13	4	car	3	0.89286685	386	448	124	166
```
From this type of storage we can identify the label of the detected object, the image frame id, the object id from a 
tracking algorithm, the confidence score and the bounding box coordinates.

I need to decide on how to separate the stages of detection and tracking.
The function that computes the detection, may receive an optional argument for tracking a detection
And also the separate generator for running the tracking algorithm without running the detection step should remain.
Maybe I change the tracking algorithm without changing the detection algorithm

If the only the detection step is done without tracking each detection, then the output of the CSV should look like this.
Another reason for running the detection step without tracking is because we may choose to skip every 5 frames.
```html
	img_id	label	score	        x1	x2	y1	y2	obj_id
0	0	car	0.957448542	63	108	118	141	
1	0	car	0.89286685	356	421	128	169	
2	0	car	0.886776984	218	274	96	125	
3	0	car	0.872502923	11	69	152	182	
4	0	truck	0.801649988	11	301	297	480	
```
How do we deal with processing every 5 frames and also obtain the object ids? I need to do some offline analysis of the
 N/5 analized frames.
 
I need to select the intervals that represents truck activity and rerun the detection over the frames in that interval. 
Once this is done the tracking algorithm can be executed and assign object ids for the frames with detections. Frames 
without detections are just simply ignored.

