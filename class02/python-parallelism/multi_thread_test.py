import threading
import os
import queue
from tqdm import tqdm
from functools import partial
#from pytube import YouTube
from pytubefix import YouTube
import time
from threading import Lock
import concurrent.futures
import multiprocessing

def say_numbers():
    # Count 1~5
    for i in range(1,6,1):
        print(f"Number: #{i}")
        time.sleep(0.1)

def say_alphabet():
    # Count a~e
    for i in range(ord("a"), ord("f")):
        print(f"Alphabet: {chr(i)}")
        time.sleep(0.1)
        
print("-"*10+"Sequential execution"+"-"*10)
seq = time.time()
say_numbers()
say_alphabet()
print("execution time :", time.time() - seq)

print("-"*10+"Parallel execution"+"-"*10)
par = time.time()   # 러닝타임 체크 시작
p1 = multiprocessing.Process(target=say_numbers)
p2 = multiprocessing.Process(target=say_alphabet)
p1.start()
p2.start()

# join으로 대기하지 않으면 부모 process가 종료되어 자식이 zombie가 된다.
p1.join()
p2.join()

print("execution time :", time.time() - par)    # 러닝타임 체크 끝

def append_one(l):
    l.append(1)

def append_two(l):
    l.append(2)

# Different threads are able to access on list address
print("-"*10+"Multi-threading"+"-"*10)
list1 = []
t1 = threading.Thread(target=append_one, args=(list1,))
t2 = threading.Thread(target=append_two, args=(list1,))
t1.start()
t2.start()
t1.join()
t2.join()
print(f"Multi-threading result:{list1}")

# Different processes can't.
print("-"*10+"Multi-processing"+"-"*10)
list2 = []
process1 = multiprocessing.Process(target=append_one, args=(list2,))
process2 = multiprocessing.Process(target=append_two, args=(list2,))
process1.start()
process2.start()
process1.join()
process2.join()
print(f"Multi-processing result:{list2}")

shared_variable = 0

def increment_shared_variable():
    global shared_variable
    for _ in range(1000000):
        shared_variable += 1

print("-"*10+"Using unsafe variable"+"-"*10)

threads = []
for _ in range(5):
    t = threading.Thread(target=increment_shared_variable)
    threads.append(t)

for t in threads:
    t.start()

for t in threads:
    t.join()

print("Final result:", shared_variable)

shared_variable = 0

# Thread-safe
lock = threading.Lock()

def increment_shared_variable():
    global shared_variable
    for _ in range(1000000):
        # Mutex lock
        lock.acquire()
        shared_variable += 1
        # Mutex release
        lock.release()

print("-"*10+"Using mutex"+"-"*10)

threads = []
for _ in range(5):
    t = threading.Thread(target=increment_shared_variable)
    threads.append(t)

for t in threads:
    t.start()

for t in threads:
    t.join()

print("Final result:", shared_variable)

def draw_ui(var):
    print("UI thread starting ... PID:{}".format(os.getpid()))
    prev = 0
    tqdm_bar = None
    while True:
        message = var.get()
        if message["type"] == "on_progress":
            if tqdm_bar is None:
                tqdm_bar = tqdm(total=100, desc="Downloading...")
            cur_rate = message["progress_rate"]
            tqdm_bar.update(int(cur_rate-prev))
            prev = int(cur_rate)
        elif message["type"] == "on_complete":
            if tqdm_bar is None:
                tqdm_bar = tqdm(total=100, desc="Downloading...")
            tqdm_bar.update(100-prev)
            tqdm_bar.close()
            break
        
def on_progress(stream, chunk, bytes_remaining, var):
    total_size = stream.filesize
    bytes_downloaded = total_size - bytes_remaining
    progress = (bytes_downloaded / total_size) * 100
    var.put({"type":"on_progress", "progress_rate":progress})

def on_complete(stream, file_handle, var):
    var.put({"type":"on_complete"})

def download(url, var):
    print("Download thread starting ... PID:{}".format(os.getpid()))
    on_progress_with_Q = partial(on_progress, var=var)
    on_complete_with_Q = partial(on_complete, var=var)
    youtube_clip = YouTube(
                        url,
                        on_progress_callback=on_progress_with_Q,
                        on_complete_callback=on_complete_with_Q)
    youtube_stream = youtube_clip.streams.get_highest_resolution()
    youtube_stream.download("videos")
    
# 코난 오브라이언
url = "https://www.youtube.com/watch?v=mAMN2ffEUBc"

print("main process running ... PID:{}".format(os.getpid()))

shared_var = queue.Queue()  # message_queue = multiprocessing.Queue()가 아니라 그냥 queue.Queue()로 사용

t1 = threading.Thread(target=draw_ui, args=(shared_var,))
t2 = threading.Thread(target=download, args=(url, shared_var,))

t1.start()
t2.start()

t1.join()
shared_var.put(None)
t2.join()

class ThreadSafeList:
    def __init__(self):
        self._list = []
        self._lock = threading.Lock()

    def append(self, item):
        with self._lock:
            self._list.append(item)

    def pop(self, index=-1):
        with self._lock:
            return self._list.pop(index)

    def __len__(self):
        with self._lock:
            return len(self._list)
        
def draw_ui(shared_queue):
    print("UI thread starting ... PID:{}".format(os.getpid()), flush=True)
    prev = 0
    tqdm_bar = None
    while True:
        if len(shared_queue)>0:
            message = shared_queue.pop()
            if message["type"] == "on_progress":
                if tqdm_bar is None:
                    tqdm_bar = tqdm(total=100, desc="Downloading...")
                cur_rate = message["progress_rate"]
                tqdm_bar.update(int(cur_rate-prev))
                prev = int(cur_rate)
            elif message["type"] == "on_complete":
                if tqdm_bar is None:
                    tqdm_bar = tqdm(total=100, desc="Downloading...")
                tqdm_bar.update(100-prev)
                tqdm_bar.close()
                break
def on_progress(stream, chunk, bytes_remaining, shared_queue):
    total_size = stream.filesize
    bytes_downloaded = total_size - bytes_remaining
    progress = (bytes_downloaded / total_size) * 100
    shared_queue.append({"type":"on_progress", "progress_rate":progress})

def on_complete(stream, file_handle, shared_queue):
    shared_queue.append({"type":"on_complete"})

def download(url, shared_queue):
    print("Download thread starting ... PID:{}".format(os.getpid()), flush=True)
    on_progress_with_Q = partial(on_progress, shared_queue=shared_queue)
    on_complete_with_Q = partial(on_complete, shared_queue=shared_queue)
    youtube_clip = YouTube(
                        url,
                        on_progress_callback=on_progress_with_Q,
                        on_complete_callback=on_complete_with_Q)
    youtube_stream = youtube_clip.streams.filter(
                        adaptive=True, 
                        file_extension='mp4').first()
    youtube_stream.download("videos")
        
# One call away
url = "https://www.youtube.com/watch?v=BxuY9FET9Y4"

print("main process running ... PID:{}".format(os.getpid()), flush=True)

shared_queue = ThreadSafeList()

t1 = threading.Thread(target=draw_ui, args=(shared_queue,))
t2 = threading.Thread(target=download, args=(url, shared_queue,))

t1.start()
t2.start()

t1.join()
t2.join()
shared_queue = None      
    