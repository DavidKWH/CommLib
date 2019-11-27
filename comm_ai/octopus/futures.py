'''
Future class
'''
import rqueues as rq

class Future:
    '''
    class that holds information about a task
    '''
    def __init__(self, task_id):
        self.task_id = task_id
        self.accepted = False
        self.canceled = False
        self.running = False
        self.done = False

    def is_canceled(self):
        return self.canceled

    def is_running(self):
        if not self.accepted:
            # check reponses from workers
            process_responses()
            qs = rq.get_queues()
            qs.from_runners()
            # set flags if we have a response
            #

        return self.running

    def is_done(self):
        return self.done

    def cancel(self):
        ''' cancel task '''
        pass

    def result(self, timeout=0):
        ''' wait on result '''
        pass

