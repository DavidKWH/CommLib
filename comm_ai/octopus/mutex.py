'''
Implement distributed mutex over storage object
'''

class Mutex:
    def __init__(self, storage, node_id, name, timeout=-1):
        pass

    def acquire(self):
        key = storage.get_token(name)
        while (key != 0):
            pass
            key = node_id
            if (key == node_id):
                break
            # perhaps yield to other threads

    def release(self):
        key = 0


