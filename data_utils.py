"""
Data util codes based on https://github.com/domluna/memn2n
"""

import os
import re
import numpy as np
import pickle
import networkx as nx
import matplotlib; from matplotlib import pyplot as plt


def get_base_node(dbca):
    for node in dbca.nodes:
        a_q_predecessors = [pred for pred in dbca.predecessors(node) if (pred[:2] == "A_" or pred[:2] == "Q_")]
        if node[:2] == "A_" and not a_q_predecessors:
            return node
    raise ValueError("no base action in graph")


def clear_node(dbca, curr_node):
    for node in list(dbca.predecessors(curr_node)):
        if dbca.degree(node) < 2:
            dbca.remove_node(node)
    dbca.remove_node(curr_node)


def decompose_sub_graph(dbca, curr_node, prev_node):
    next_node = list(dbca.successors(curr_node))[0]
    dbca.remove_edge(curr_node, next_node)

    sub_graph = nx.subgraph(dbca, [node for node in dbca if nx.has_path(dbca, node, curr_node)]).copy()

    clear_node(dbca, curr_node)
    dbca.add_edge(prev_node, next_node)

    return sub_graph, prev_node


def decompose_dbca(dbca_list):
    sub_graphs = []
    for dbca in dbca_list:
        curr_node = get_base_node(dbca)
        prev_node = None
        while True:
            if not dbca[curr_node]:
                sub_graph = nx.subgraph(dbca, dbca.nodes)
                sub_graphs.append(sub_graph)
                break
            if curr_node[:2] == "Q_":
                sub_graph, curr_node = decompose_sub_graph(dbca, curr_node, prev_node)
                sub_graphs.append(sub_graph)
            prev_node = curr_node
            curr_node = list(dbca[curr_node])[0]
    return sub_graphs


def load_dbca(data_dir, task_id, valid=True):
    import sys
    sys.path.insert(0, '/home/aviad/PycharmProjects/babi_generator')
    import babi_generator

    files = os.listdir(data_dir)
    files = [os.path.join(data_dir, f) for f in files]
    s = "dbca{}_".format(task_id)
    eval_str = "_valid" if valid else "_test"
    train_file = [f for f in files if s in f and 'train' in f][0]
    test_file = [f for f in files if s in f and eval_str in f][0]

    with open(train_file, 'rb') as train_file:
        dbca_list = pickle.load(train_file)
        train_graphs = decompose_dbca(dbca_list)
    with open(test_file, 'rb') as test_file:
        dbca_list = pickle.load(test_file)
        test_graphs = decompose_dbca(dbca_list)

    return train_graphs, test_graphs

def load_data_wandb(wandb_run, wandb_path):
    artifact = wandb_run.use_artifact(wandb_path, type='dataset')
    data_dir = artifact.download()
    return data_dir

def load_task(data_dir, task_id, only_supporting=False, valid=True, load_wandb=True, wandb_run=None):
    """
    Load the nth task. There are 20 tasks in total.
    Returns a tuple containing the training and testing data for the task.
    """
    # assert task_id > 0 and task_id < 21
    files = os.listdir(data_dir)
    files = [os.path.join(data_dir, f) for f in files]
    s = "qa{}_".format(task_id)
    eval_str = "_valid" if valid else "_test"
    train_file = [f for f in files if s in f and 'train' in f][0]
    test_file = [f for f in files if s in f and eval_str in f][0]
    train_data = get_stories(train_file, only_supporting)
    test_data = get_stories(test_file, only_supporting)
    return train_data, test_data


def get_tokenized_text(data_dir, task_id):
    # assert task_id > 0 and task_id < 21
    files = os.listdir(data_dir)
    files = [os.path.join(data_dir, f) for f in files]
    s = "qa{}_".format(task_id)
    train_file = [f for f in files if s in f and 'train' in f][0]
    valid_file = [f for f in files if s in f and "_valid" in f][0]
    test_file = [f for f in files if s in f and "_test" in f][0]

    train_text, valid_text, test_text = "", "", ""
    with open(train_file) as train_file:
        lines = train_file.readlines()
        for line in lines:
            train_text += line;
        train_text = str.lower(train_text)
        train_text = tokenize(train_text)
    with open(valid_file) as valid_file:
        lines = valid_file.readlines()
        for line in lines:
            valid_text += line;
        valid_text = str.lower(valid_text)
        valid_text = tokenize(valid_text)
    with open(test_file) as test_file:
        lines = test_file.readlines()
        for line in lines:
            test_text += line;
        test_text = str.lower(test_text)
        test_text = tokenize(test_text)

    return train_text + valid_text + test_text

def tokenize(sent):
    """
    Return the tokens of a sentence including punctuation.
    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    """
    return [x.strip() for x in re.split("(\W+)", sent) if x.strip()]


def parse_stories(lines, only_supporting=False):
    """
    Parse stories provided in the bAbI tasks format
    If only_supporting is true, only the sentences that support the answer are kept.
    """
    data = []
    story = []
    for line in lines:
        # hack to ignore empty lines at end
        if len(line) < 2:
            continue
        try:
            line = str.lower(line)
            nid, line = line.split(" ", 1)
        except Exception as e:
            print("Line is ", line)
            print(str(e))
            
        nid = int(nid)
        if nid == 1:
            story = []
        if "\t" in line: # question
            q, a = line.split("\t")[0], line.split("\t")[1]
            q = tokenize(q)
            # a = tokenize(a)
            # answer is one vocab word even if it's actually multiple words
            a = [str.rstrip(a)]
            substory = None

            # remove question marks
            if q[-1] == "?":
                q = q[:-1]

            if only_supporting:
                # Only select the related substory
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:
                # Provide all the substories
                substory = [x for x in story if x]

            data.append((substory, q, a))
            story.append("")
        else: # regular sentence
            # remove periods
            sent = tokenize(line)
            if sent[-1] == ".":
                sent = sent[:-1]
            story.append(sent)
    return data


def get_stories(f, only_supporting=False):
    """
    Given a file name, read the file, retrieve the stories,
    and then convert the sentences into a single story.
    If max_length is supplied, any stories longer than max_length
    tokens will be discarded.
    """
    with open(f) as f:
        return parse_stories(f.readlines(), only_supporting=only_supporting)


def vectorize_data(data, word_idx, sentence_size, memory_size):
    """
    Vectorize stories and queries.
    If a sentence length < sentence_size, the sentence will be padded with 0's.
    If a story length < memory_size, the story will be padded with empty memories.
    Empty memories are 1-D arrays of length sentence_size filled with 0's.
    The answer array is returned as a one-hot encoding.
    """
    S, Q, A = [], [], []
    for story, query, answer in data:
        ss = []
        for i, sentence in enumerate(story, 1):
            ls = max(0, sentence_size - len(sentence))
            ss.append([word_idx[w] for w in sentence] + [0] * ls)

        # take only the most recent sentences that fit in memory
        ss = ss[::-1][:memory_size][::-1]

        # Make the last word of each sentence the time 'word' which
        # corresponds to vector of lookup table
        for i in range(len(ss)):
            ss[i][-1] = len(word_idx) - memory_size - i + len(ss)

        # pad to memory_size
        lm = max(0, memory_size - len(ss))
        for _ in range(lm):
            ss.append([0] * sentence_size)

        lq = max(0, sentence_size - len(query))
        q = [word_idx[w] for w in query] + [0] * lq

        y = np.zeros(len(word_idx) + 1) # 0 is reserved for nil word
        for a in answer:
            y[word_idx[a]] = 1

        S.append(ss); Q.append(q); A.append(y)
    return np.array(S), np.array(Q), np.array(A)


# load_dbca("./../babi_data/tasks_1-20_v1-2/en-valid-10k", 29)