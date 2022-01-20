import threading, time
# MyThread.py :  Thread Class
class MyThread(threading.Thread):
    def __init__(self, func, args=()):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args
 
    def run(self):
        self.result = self.func(*self.args)
 
    def get_result(self):
        threading.Thread.join(self) # Wait for the thread to finish executing
        try:
            return self.result
        except Exception:
            return None