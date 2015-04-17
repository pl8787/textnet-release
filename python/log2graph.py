import sys
import re

reduce_interval = {
    "Train" : 100,
    "Valid" : 100,
    "Test" : 100
        }

seperator = ','  # '\t'

def reduce_result(x, interval):
    y = {}
    count = {}
    for iter, value in x:
        idx = iter / interval
        if idx not in y:
            y[idx] = 0.0
            count[idx] = 0
        y[idx] += value
        count[idx] += 1
    for idx in y:
        y[idx] /= count[idx]

    # convert dict to list
    idx = 0
    rtn = []
    while True:
        if idx in y:
            rtn.append(y[idx])
            idx += 1
        else:
            break
    return rtn

# test = '[Train:kTrain]\tIter\t3321:\tOut[loss] =\t0.228255'
pattern_raw = r"\[(.+?)\:(.+?)\]\tIter\t(.+?)\:\tOut\[(.+?)\].=\t(.+)"

log_lines = {}

for line in open(sys.argv[1]):
    m = re.match(pattern_raw, line)
    if m:
        tag = m.group(1)
        phase = m.group(2)
        iter = int(m.group(3))
        node = m.group(4)
        value = float(m.group(5))
        if tag not in log_lines:
            log_lines[tag] = {}
        if node not in log_lines[tag]:
            log_lines[tag][node] = []
        log_lines[tag][node].append([iter, value])

for tag in log_lines:
    for node in log_lines[tag]:
        log_lines[tag][node] = reduce_result(log_lines[tag][node], reduce_interval[tag])
        print len(log_lines[tag][node])

_out = open(sys.argv[1]+".csv", 'w')
# Write header
_out.write("iter%s" % seperator)
for tag in log_lines:
    for node in log_lines[tag]:
        _out.write("%s:%s%s"%(tag, node, seperator))
_out.write("\n")

idx = 0
while True:
    have_data = False
    _out.write("%d%s"%(idx, seperator))
    for tag in log_lines:
        for node in log_lines[tag]:
            if idx < len(log_lines[tag][node]):
                _out.write("%f%s"%(log_lines[tag][node][idx], seperator))
                have_data = True
            else:
                _out.write(seperator)
    _out.write("\n")
    if not have_data:
        break
    idx += 1

_out.close()


