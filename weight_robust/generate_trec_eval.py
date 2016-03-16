import json
import sys

# Read Result
def read_result(filename):
	json_str = open(filename).read()
	json_obj = json.loads(json_str)
	score = []
	for d in json_obj:
		score.append( (d['fc2']['data']['value'], d['label']['data']['value']) )
	return score

pair_file = open('/home/pangliang/matching/data/robust/data/relation.test.fold0.txt')
gt_file = open('/home/pangliang/matching/data/robust/origin/robust04.qrels')

# Read Pair
pairs = []
for line in pair_file:
	pairs.append(line.strip().split())
print len(pairs)

# Read GT
qid_gt = {}
for line in gt_file:
	part = line.strip().split()
	qid = part[0]
	if qid not in qid_gt:
		qid_gt[qid] = []
	qid_gt[qid].append(line.strip())
print len(qid_gt)

score = read_result(sys.argv[1])
tag = sys.argv[2]
res_file = open('trecResFile_%s.txt' % tag, 'w')
rel_file = open('trecRelGTFile_%s.txt' % tag, 'w')

base_idx = 0

for Q_list in score:
	Q_list = [ (pairs[base_idx + idx], x[0], x[1]) for idx, x in enumerate(zip(*Q_list)) ]

	base_idx += len(Q_list)
	print base_idx

	# Check Valid
	qid = ''
	for s in Q_list:
		if not qid:
			qid = s[0][1]
		else:
			assert qid == s[0][1]

	# Write Res File
	Q_list = sorted(Q_list, key = lambda x : x[1], reverse = True)
	for rank, s in enumerate(Q_list):
		print >>res_file, '\t'.join( map(str, [s[0][1], 0, s[0][2], rank+1, s[1], 'CNN']) )

	# Write Rel File
	for line in qid_gt[qid]:
		print >>rel_file, line

res_file.close()
rel_file.close()
