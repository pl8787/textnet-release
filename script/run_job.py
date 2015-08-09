#-*-coding:utf8-*-
import sys, time, traceback, random, os, json

from threading import Thread
from subprocess import *
from Queue import Queue

def run_cmd(cmd):
    print cmd
    p = Popen(cmd,shell=True)
    p.wait()
    return 

# cmd upload local files to remote files
def scp2remote(srcFile, node, destFile):
    cmd = 'scp {0} {1}:{2}'.format(srcFile, node.get_user_at_ip(), destFile)
    print cmd
    p = Popen(cmd,shell=True)
    p.wait()

# cmd download remote files to local files
def scp2local(node, srcFile, destFile):
    cmd = 'scp {0}:{1} {2}'.format(node.get_user_at_ip(), srcFile, destFile)
    print cmd
    p = Popen(cmd,shell=True)
    p.wait()

# cmd
def mkdir(node, path):
    cmd = 'ssh -x -t {0} "mkdir -p {1}"'.format(node.get_user_at_ip(), path)
    p = Popen(cmd,shell=True)
    p.wait()

def get_proc_ids(node, procKeywords):
    cmd = 'ssh -x -t {0} "ps a -u wsx"'.format(node.get_user_at_ip())
    print cmd
    proc_ids = []
    ret = Popen(cmd,shell=True,stdout=PIPE,stderr=PIPE,stdin=PIPE).stdout
    for line in ret.readlines():
        if 'ssh' in line or 'bash' in line:
            continue
        is_include_all = True 
        for w in procKeywords:
            if not w in line:
                is_include_all = False
                break
        if is_include_all:
            proc_ids.append(int(line.strip().split(' ')[0]))
    return proc_ids 



# 检查一个节点是否是free的，主要看下面进程数有没有达到max num
def is_node_free(node, procKeywords, procMaxNum):
    procs = get_proc_ids(node, procKeywords)
    if len(procs) < procMaxNum:
        return True
    else:
        return False

# 杀掉所有包含关键词的进程
def kill_job(node, procKeywords):
    procs = get_proc_ids(node, procKeywords);
    for proc in procs:
        cmd = 'ssh -x -t {0} "kill {1}"'.format(node.get_user_at_ip(), str(proc))
        p = Popen(cmd,shell=True)
        p.wait()
    return

class WorkerStopToken:  # used to notify the worker to stop or if a worker is dead
    pass

class Node:
    def __init__(self, ip, user, num_proc):
        self.ip = ip
        self.user = user
        self.num_proc = num_proc
    def get_user_at_ip(self):
        return self.user + '@' + self.ip

class Job:
    def __init__(self, job_id, bin, local_work_dir, remote_work_dir, conf_file, log_file):
        self.job_id = job_id
        self.conf_file = conf_file
        self.log_file = log_file
        self.remote_work_dir = remote_work_dir
        self.local_work_dir = local_work_dir
        self.bin = bin

    # for identifying this job 
    def gen_identity_str_set(self):
        return [self.bin, self.conf_file]

    def local_conf_file(self):
        return self.local_work_dir + self.conf_file
    def local_log_file(self):
        return self.local_work_dir + self.log_file
    def remote_conf_file(self):
        return self.remote_work_dir + self.conf_file
    def remote_log_file(self):
        return self.remote_work_dir + self.log_file

class SshWorker(Thread):
    def __init__(self,name,node,dev,job_queue,options):
        Thread.__init__(self)
        self.name = name 
        self.node = node
        self.dev = dev
        self.job_queue = job_queue
        self.options = options # max_proc_num 
        assert self.dev == 'cpu' or self.dev == 'gpu'
        
    def run(self):
        while True:
            # 防止所有线程同时启动，同时检查到系统free，然后开太多进程
            time.sleep(random.randint(1,100))
            isDone = False
            while True:
                job = self.job_queue.get()
                if job is WorkerStopToken:
                    self.job_queue.put(job)
                    print('all job done, worker {0} stop.'.format(self.name))
                    isDone = True
                    break
                if not is_node_free(self.node, [job.bin], self.options['max_proc_num']):
                    print '{0}: is waiting job begin...'.format(self.name)
                    self.job_queue.put(job)
                    time.sleep(600)
                else:
                    break

            if isDone:
                break

            try:
                p = self.run_one(job)
            except:
                # # we failed, let others do that and we just quit
                # we failed, do it again
                traceback.print_exception(sys.exc_info()[0], sys.exc_info()[1], sys.exc_info()[2])
                self.job_queue.put(job)
                # sys.stderr.write('worker {0} fail and quit.\n'.format(self.name))
                # break
            
            time.sleep(10)
            while not self.is_job_done(job.gen_identity_str_set()):
                print '{0}: is waiting job end...'.format(self.name)
                time.sleep(30)
            self.get_job_result(job)
            time.sleep(10)

    def run_one(self, job):
        scp2remote(job.local_conf_file(), self.node, job.remote_conf_file())
        cmd = 'ssh -x -t -f {0} "cd {1};./{2} {3}"'.\
              format(self.node.get_user_at_ip(), \
                     job.remote_work_dir, \
                     job.bin, \
                     job.conf_file)
        print cmd
        p = Popen(cmd,shell=True)
        return p

    def is_job_done(self, procKeywords):
        return is_node_free(self.node, procKeywords, 1)

    # 这个函数取回运行结果
    def get_job_result(self, job):
        scp2local(self.node, job.remote_log_file(), job.local_log_file())

def get_nodes():
    n_thread = 10
    node_169 = Node('10.60.1.169', 'wsx', n_thread/2)
    node_52 = Node('10.60.0.52', 'wsx', n_thread)
    node_59 = Node('10.60.0.59', 'wsx', n_thread)
    # node_168 = Node('10.60.1.168', 'wsx', 8)
    return [node_52, node_59]
    # return [node_169]
    # return [node_52, node_59]
    # return [node_59]

def main():
    run_nodes = get_nodes()
    # kill_job(run_nodes[0] , ['textnet'])
    # kill_job(run_nodes[1] , ['textnet'])
    # kill_job(run_nodes[2] , ['textnet'])
    # exit(0)

    # max_proc_num = sys.args[1]
    # bin = sys.args[2]
    # local_dir = sys.args[3]
    # remote_dir = sys.args[4]
    bin = 'textnet'
    # local_dir = '/home/wsx/exp/topk_simulation/run.4/'
    # local_dir = '/home/wsx/exp/gate/lstm/run.9/'
    # local_dir  = '/home/wsx/exp/match/birnn_mlp/run.1/'
    # local_dir = '/home/wsx/exp/match/msrp/cnn/run.2/'
    # local_dir = '/home/wsx/exp/ccir2015/mr/bilstm/run.2/'
    # local_dir = '/home/wsx/exp/ccir2015/mr/birnn/run.2/'
    # local_dir = '/home/wsx/exp/ccir2015/mr/birnn/run.2/'
    # local_dir = '/home/wsx/exp/match/msrp_char/bilstm_sim_dpool/run.3/'
    # local_dir = '/home/wsx/exp/match/msrp_dpool/run.1/'
    # local_dir = '/home/wsx/exp/match/msrp/lstm_sim_dpool/run.29/'
    #local_dir = '/home/wsx/exp/match/msrp/lstm_sim_dpool/run.30/'
    local_dir = '/home/wsx/exp/match/qa_balance/bilstm_sim_dpool/run.3/'
    # local_dir = '/home/wsx/exp/nbp/tf/run.8/'
    remote_dir = '/home/wsx/log.tmp/'

    conf_files = os.listdir(local_dir) 
    print conf_files

    jobQue = Queue(0)
    for i, conf_file in enumerate(conf_files):
        if 'cfg' not in conf_file and 'model' not in conf_file:
            continue
        conf = json.loads(open(local_dir+conf_file).read())
        log_file = conf['log']
        job = Job(i, bin, local_dir, remote_dir, conf_file, log_file)
        jobQue.put(job)
    jobQue.put(WorkerStopToken)

    worker_id = 0
    for node in run_nodes:
        for proc in range(node.num_proc):
            worker = SshWorker('worker_'+str(worker_id), node, 'cpu', jobQue, {'max_proc_num':node.num_proc})
            print 'start worker:', worker_id
            worker.start()
            worker_id += 1

if __name__ == '__main__':
    main()
