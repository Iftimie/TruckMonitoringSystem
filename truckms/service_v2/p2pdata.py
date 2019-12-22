import montydb


def p2p_insert_one(db, col, data, nodes=None):
    if nodes is None:
        nodes = []
    collection = montydb.MontyClient[db][col]
    data["nodes"] = nodes
    collection.insert_one(data)


def p2p_update(db, col, filter, update):
    collection = montydb.MontyClient[db][col]
    collection.update_one(filter, update)


import ray
ray.init()

@ray.remote
def f(x):
    return x * x

futures = [f.remote(i) for i in range(4)]
print(ray.get(futures))


# mycol.create_index({"name": "name", "unique": True})
