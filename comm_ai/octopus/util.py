'''
helper functions
'''
import os
import uuid
import socket
import functools
from datetime import datetime
from hashlib import blake2b

#def task_id():
#    return node_id()

NUM_RAND_BITS = 16
DIGEST_SIZE = 16

def message_id():
    return node_id()

def node_id():
    '''
    generate node_id from the following sources

     * uuid MAC addresses
     * current UTC time
     * hostname
     * urandom
    '''
    mac_addr = uuid.getnode()
    now = datetime.utcnow()
    host_id = socket.gethostname()
    rnd_num = os.urandom(NUM_RAND_BITS)

    h = blake2b(digest_size=DIGEST_SIZE)
    h.update(f'{mac_addr:x}'.encode())
    h.update(f'{now}'.encode())
    h.update(host_id.encode())
    h.update(rnd_num)
    return h.hexdigest()

def hash_fn(buf):
    '''
    hash byte buffer for consistency check
    '''
    h = blake2b(digest_size=DIGEST_SIZE)
    h.update(buf)
    return h.hexdigest()

################################################################################
# custom retry decorator
################################################################################
def conn_retry(func):
    ''' custom retry decorator '''
    MAX_RETRIES = 3
    EXCEPTION = BrokenPipeError

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        for i in range(MAX_RETRIES):
            #print('attempt', i)
            failed = False
            try:
                value = func(*args, **kwargs)
            except EXCEPTION as e:
                #print('caught exception', type(e))
                failed = True
                if i+1 == MAX_RETRIES:
                    raise RuntimeError('Maximum retries reached') from e
            if not failed:
                #print('all good')
                break
        # return function value
        return value

    return wrapper

