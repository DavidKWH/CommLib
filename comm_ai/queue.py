'''
Queue over Redis
'''
import redis
import psutil
import subprocess
from subprocess import CalledProcessError
from subprocess import TimeoutExpired
# use hotqueue implementation
# See https://github.com/richardhenry/hotqueue
from hotqueue import HotQueue as RedisQueue

################################################################################
# module initialization
################################################################################
#pname = "redis-server"
#if pname not in (p.name() for p in psutil.process_iter()):
#    print(f'starting {pname}')
#    proc = subprocess.Popen([pname])
#    try:
#        proc.wait(timeout=1)
#    except TimeoutExpired:
#        # harmless since we are making sure
#        # the server process did not terminate
#        # prematurely
#        pass
#    if proc.returncode is not None:
#        print(f'returncode = {proc.returncode}')
#        raise RuntimeError('server did not start...')
#else:
#    print(f'{pname} process exists')

################################################################################
# queue class
################################################################################
#class RedisQueue(object):
#    """Simple Queue with Redis Backend"""
#    def __init__(self, name, namespace='queue', **redis_kwargs):
#        """The default connection parameters are: host='localhost', port=6379, db=0"""
#        self.__db= redis.Redis(**redis_kwargs)
#        self.key = '%s:%s' %(namespace, name)
#
#    def qsize(self):
#        """Return the approximate size of the queue."""
#        return self.__db.llen(self.key)
#
#    def empty(self):
#        """Return True if the queue is empty, False otherwise."""
#        return self.qsize() == 0
#
#    def put(self, item):
#        """Put item into the queue."""
#        self.__db.rpush(self.key, item)
#
#    def get(self, block=True, timeout=None):
#        """Remove and return an item from the queue.
#
#        If optional args block is true and timeout is None (the default), block
#        if necessary until an item is available."""
#        if block:
#            item = self.__db.blpop(self.key, timeout=timeout)
#        else:
#            item = self.__db.lpop(self.key)
#
#        if item:
#            item = item[1]
#        return item
#
#    def get_nowait(self):
#        """Equivalent to get(False)."""
#        return self.get(False)
