from multiprocessing.pool import ThreadPool
pool = ThreadPool(processes=1)



def r(k):
	return k+1


for x in range(1000):

	async_result = pool.apply_async(r,[x])
	p = async_result.get()
	print(p)

