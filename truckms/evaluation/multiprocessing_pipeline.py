from multiprocessing import Pool
import time

def transform_func(x):
	time.sleep(0.5)
	return -x

def generator():
	for i in range(100):
		yield i

def generator2(g):
	for i in g:
		yield i + 0.1

def generator_final(g):
	for i in g:
		yield i + 0.0001

if __name__ == "__main__":
	pool = Pool(processes=10)
	res_gen = pool.imap_unordered(transform_func, generator2(generator()))
	final_gen = generator_final(res_gen)
	for i in final_gen:
		print (i)