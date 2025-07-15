import redis
import yaml
from rediskeys import RedisKeys
import json
import numpy as np

redis_client = redis.Redis()

with open("config.yaml", "r") as file:
    data = yaml.safe_load(file)

print(data)

default_cart_pos = [0,0,0]
default_cart_orient = [1,0,0,0]

def init_redis():
    for i in range(data["num_envs"]):
        redis_client.set("env{}::{}".format(i, RedisKeys.DES_CART_POS.value), json.dumps(default_cart_pos))
        redis_client.set("env{}::{}".format(i, RedisKeys.DES_CART_ORIENT.value), json.dumps(default_cart_orient))

init_redis()


