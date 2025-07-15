from enum import Enum

class RedisKeys(Enum):
    CART_POS = "cart_pos"
    CART_ORIENT = "cart_orient"
    GRIP_POS = "grip_pos"

    DES_CART_POS = "des_cart_pos"
    DES_CART_ORIENT = "des_cart_orient"
    DES_GRIP_POS = "des_grip_pos"