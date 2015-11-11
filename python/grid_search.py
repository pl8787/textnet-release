import sys
import os
import json
import copy
import multiprocessing

if len(sys.argv) < 5:
    print "Usage: python grid_search.py [model_template_file] [parameters_config] [process_number] [run_dir]"

model_tpl_file = sys.argv[1]
param_conf_file = sys.argv[2]
proc_num = int(sys.argv[3])
run_dir = sys.argv[4]

if not os.path.isdir(run_dir):
    os.mkdir(run_dir)
    print 'Create Run Directory'

model_template = open(model_tpl_file).read()
param_config = {}
for line in open(sys.argv[2]):
    line = line.strip().split()
    param_config[line[0]] = line[1:]
print 'Read Template & Config over.'

def render_template(template, params):
    for k, v in params.items():
        template = template.replace('{{%s}}' % k, v)
    return template

_p = [ [0, k, 0, len(v)] for k, v in param_config.items() ]
def get_one_config(p, d):
    rtn = []
    if d == len(p):
        return [{k:param_config[k][idx] for idx, k, _, __ in p}]
    for i in range(p[d][3]):
        rtn += get_one_config(copy.deepcopy(p), d+1)
        p[d][0] += 1
    return rtn

models_list = []
config_out_file = open(run_dir + '/run.conf', 'w')
for idx, config in enumerate(get_one_config(_p, 0)):
    model = render_template(model_template, config)
    try:
        obj = json.loads(model)
    except Exception as e:
        print e
        exit()
    model_file = run_dir + '/' + model_tpl_file.split('/')[-1] + '.run%d' % idx
    log_file = run_dir + '/' + model_tpl_file.split('/')[-1] + '.log%d' % idx
    print model_file
    print >>config_out_file, model_file, config
    open(model_file, 'w').write(model)
    models_list.append((model_file, log_file))
config_out_file.close()

def run_one_model(model_path, log_path):
    BIN = '/home/pangliang/matching/textnet_statistic/bin/textnet'
    command = '%s %s >%s 2>&1' % (BIN, model_path, log_path)
    print command
    os.system(command)

# Schedule 
pool = multiprocessing.Pool(processes = proc_num)
for model_path, log_path in models_list:
    pool.apply_async(run_one_model, (model_path, log_path))
pool.close()
pool.join()


