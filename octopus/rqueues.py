'''
Implement queues for task communications
'''
from .queue import RedisQueue

class TaskQueues:
    def __init__(self, **kwargs):
        '''
        Accepts kwargs for Redis connection
        '''
        # create queues
        self.to_runners = RedisQueue('to_runners', **kwargs)
        self.from_runners = RedisQueue('from_runners', **kwargs)

