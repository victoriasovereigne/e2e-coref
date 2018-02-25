#!/usr/bin/env python

import re
import os
import sys
import json
import tempfile
import subprocess
import collections

import util
import conll
# from nltk.tag import StanfordNERTagger

class DocumentState(object):
  def __init__(self):
    self.doc_key = None
    self.text = []
    self.text_speakers = []
    self.speakers = []
    self.sentences = []
    self.clusters = collections.defaultdict(list)
    self.stacks = collections.defaultdict(list)

    # I added this
    self.pos_tags = [] # list of pos tags, should be equivalent to self.sentences
    self.text_pos_tags = [] # should be equivalent to self.text
    self.ner_tags = []
    self.text_ner_tags = []
    self.ner_stack = []

  def assert_empty(self):
    assert self.doc_key is None
    assert len(self.text) == 0
    assert len(self.text_speakers) == 0
    assert len(self.sentences) == 0
    assert len(self.speakers) == 0
    assert len(self.clusters) == 0
    assert len(self.stacks) == 0

  def assert_finalizable(self):
    assert self.doc_key is not None
    assert len(self.text) == 0
    assert len(self.text_speakers) == 0
    assert len(self.sentences) > 0
    assert len(self.speakers) > 0
    assert all(len(s) == 0 for s in self.stacks.values())

  def finalize(self):
    merged_clusters = []
    for c1 in self.clusters.values():
      existing = None
      for m in c1:
        for c2 in merged_clusters:
          if m in c2:
            existing = c2
            break
        if existing is not None:
          break
      if existing is not None:
        print("Merging clusters (shouldn't happen very often.)")
        existing.update(c1)
      else:
        merged_clusters.append(set(c1))
    merged_clusters = [list(c) for c in merged_clusters]
    all_mentions = util.flatten(merged_clusters)
    assert len(all_mentions) == len(set(all_mentions))

    return {
      "doc_key": self.doc_key,
      "sentences": self.sentences,
      "speakers": self.speakers,
      "clusters": merged_clusters,
      "pos_tags": self.pos_tags, # I added this
      "ner_tags": self.ner_tags
    }

def normalize_word(word):
  if word == "/." or word == "/?":
    return word[1:]
  else:
    return word

def handle_line(line, document_state):
  begin_document_match = re.match(conll.BEGIN_DOCUMENT_REGEX, line)
  if begin_document_match:
    document_state.assert_empty()
    document_state.doc_key = conll.get_doc_key(begin_document_match.group(1), begin_document_match.group(2))
    return None
  elif line.startswith("#end document"):
    document_state.assert_finalizable()
    return document_state.finalize()
  else:
    row = line.split()
    if len(row) == 0:
      document_state.sentences.append(tuple(document_state.text))

      # print document_state.text
      # print document_state.text_ner_tags

      # I added this
      document_state.pos_tags.append(tuple(document_state.text_pos_tags))
      del document_state.text_pos_tags[:]

      document_state.ner_tags.append(tuple(document_state.text_ner_tags))
      del document_state.text_ner_tags[:]      

      del document_state.text[:]
      document_state.speakers.append(tuple(document_state.text_speakers))
      del document_state.text_speakers[:]

      return None
    
    # print line 
    assert len(row) >= 12

    word = normalize_word(row[3])
    coref = row[-1]
    doc_key = conll.get_doc_key(row[0], row[1])
    # print doc_key
    speaker = row[9]

    # I added this
    pos_tag = row[4]
    document_state.text_pos_tags.append(pos_tag)

    # -------------------------------------------
    # NER stuff
    ner_tag = row[10]
    ner = ''

    if ner_tag[0] == "(": # beginning of tag
      if ner_tag[-1] == ")": # one liner
        ner = "B-" + ner_tag[1:-1]
      else: 
        ner = "B-" + ner_tag[1:-1]
        document_state.ner_stack.append(ner_tag[1:-1])
    else:
      if len(document_state.ner_stack) > 0:
        ner = "I-" + document_state.ner_stack[-1]
        if ner_tag[-1] == ")":
          document_state.ner_stack.pop()
      else:
        ner = "O"

    document_state.text_ner_tags.append(ner)
    # -------------------------------------------     


    word_index = len(document_state.text) + sum(len(s) for s in document_state.sentences)
    document_state.text.append(word)
    document_state.text_speakers.append(speaker)

    if coref == "-":
      return None

    for segment in coref.split("|"):     
      if segment[0] == "(":
        if segment[-1] == ")":
          cluster_id = int(segment[1:-1])
          document_state.clusters[cluster_id].append((word_index, word_index))
        else:
          cluster_id = int(segment[1:])
          document_state.stacks[cluster_id].append(word_index)
      else:
        cluster_id = int(segment[:-1])
        start = document_state.stacks[cluster_id].pop()
        document_state.clusters[cluster_id].append((start, word_index))
    return None

def minimize_partition(name, language, extension):
  input_path = "{}.{}.{}".format(name, language, extension)
  output_path = "{}.{}.jsonlines".format(name, language)
  count = 0
  print "Minimizing {}".format(input_path)
  with open(input_path, "r") as input_file:
    with open(output_path, "w") as output_file:
      document_state = DocumentState()
      for line in input_file.readlines():
        # print input_path
        document = handle_line(line, document_state)

        if document is not None:
          output_file.write(json.dumps(document))
          output_file.write("\n")
          count += 1
          document_state = DocumentState()
  print "Wrote {} documents to {}".format(count, output_path)

def minimize_language(language):
  minimize_partition("dev", language, "v4_auto_conll")
  minimize_partition("dev2", language, "v4_auto_conll")
  minimize_partition("train", language, "v4_auto_conll")
  minimize_partition("test", language, "v4_gold_conll")

if __name__ == "__main__":
  minimize_language("english")
