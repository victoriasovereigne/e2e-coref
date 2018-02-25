gold = '00_orig/test.english.v4_gold_conll'
auto = '00_orig/test.english.v9_auto_conll'

g = open(gold, 'r')
a = open(auto, 'r')

newa = open(auto+'_2', 'w')

g_lines = g.readlines()
a_lines = a.readlines()

for i in range(len(g_lines)):
	gline = g_lines[i]
	aline = a_lines[i]

	gvals = gline.split()
	avals = aline.split()

	if len(gvals) > 10:
		coref = gvals[-1].strip()
		avals[-1] = coref

		text = aline[:-2] #'\t'.join(avals)
		newa.write(text + coref + '\n')
	else:
		newa.write(aline)

newa.close()
