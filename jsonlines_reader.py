import json
import sys
from nltk.tag import StanfordNERTagger

reload(sys)  
sys.setdefaultencoding('utf8')
# predicted = sys.argv[1] #'output1.jsonlines'

def add_NER(jsonlines, output):
	f = open(jsonlines)
	lines = f.readlines()
	ner_tagger = StanfordNERTagger('/scratch/cluster/vlestari/stanford-ner-2017-06-09/classifiers/english.all.3class.distsim.crf.ser.gz', 
                                    '/scratch/cluster/vlestari/stanford-ner-2017-06-09/stanford-ner.jar',
                                    encoding='utf-8')
	fw = open(output, 'w')
	for line in lines:
		data = json.loads(line)
		sentences = data['sentences']

		data['ner_tag'] = []

		for sentence in sentences:
			print sentence
			ner = ner_tagger.tag(sentence)
			ner = [n[1] for n in ner]
			data['ner_tag'].append(ner)
	
		fw.write(json.dumps(data)+'\n')
	fw.close()


def make_mini_dataset(filename, output, size):
	f = open(filename)
	lines = f.readlines()

	fw = open(output, 'w')

	for line in lines[:size]:
		data = json.loads(line)
		fw.write(json.dumps(data) + '\n')

	fw.close()


def read_jsonlines(predicted):
	f = open(predicted)
	lines = f.readlines()

	for line in lines:
		data = json.loads(line)
		
		sentences = data['sentences']
		gold_clusters = data['clusters']
		pred_clusters = data['predicted_clusters']
		doc_key = data['doc_key']

		word_num = 0
		doc = []
		doc2 = []

		word_clusters_gold = {}
		word_clusters_pred = {}
		
		for i in xrange(len(sentences)):
			doc.extend(sentences[i])
			doc2.extend(sentences[i])

		for cluster_num, cluster in enumerate(gold_clusters):
			for span in cluster:
				start = int(span[0])
				end = int(span[1])

				word = ' '.join(doc[start:end+1])

				if cluster_num in word_clusters_gold.keys():
					word_clusters_gold[cluster_num].append(word)
				else:
					word_clusters_gold[cluster_num] = [word]

				doc2[start] = '[' + doc2[start]
				doc2[end] = doc2[end] + ']-(' + str(cluster_num) + ')'

		for cluster_num, cluster in enumerate(pred_clusters):
			for span in cluster:
				start = int(span[0])
				end = int(span[1])

				word = ' '.join(doc[start:end+1])

				if cluster_num in word_clusters_pred.keys():
					word_clusters_pred[cluster_num].append(word)
				else:
					word_clusters_pred[cluster_num] = [word]

				doc2[start] = '{' + doc2[start]
				doc2[end] = doc2[end] + '}-(' + str(cluster_num) + ')'


		sentences_reconstructed = str(' '.join(doc2).encode("utf-8")).split(' . ')


		print '=============================================================='
		print doc_key
		print '=============================================================='
		for sentence in sentences_reconstructed:
			print sentence

		print '\n'
		print 'Gold clusters'
		for key in word_clusters_gold.keys():
			print key, ' --> ', [str(s) for s in word_clusters_gold[key]]
		
		print '\n'
		print 'Predicted clusters'
		for key in word_clusters_pred.keys():
			print key, ' --> ', [str(s) for s in word_clusters_pred[key]]

		print '\n'
		# print '=============================================================='

def main():
	method = sys.argv[1]
	print method
	if method == 'add_NER':
		add_NER(sys.argv[2], sys.argv[3])
	elif method == 'mini':
		make_mini_dataset(sys.argv[2], sys.argv[3], 10)
	else:
		read_jsonlines(sys.argv[2])