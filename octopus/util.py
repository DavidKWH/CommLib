'''
helper functions
'''
import os
import uuid
import socket
from datetime import datetime
from hashlib import blake2b

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
    rnd_num = os.urandom(16)

    h = blake2b(digest_size=32)
    h.update(f'{mac_addr:x}'.encode())
    h.update(f'{now}'.encode())
    h.update(host_id.encode())
    h.update(rnd_num)
    return h.hexdigest()

