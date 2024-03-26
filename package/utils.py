# -*- coding: utf-8 -*-
import json


def read_clue_json(path):
    tokens, labels = [], []
        tmp_tok, tmp_lab = [], []
        label_set = []
        lines = []
    
        with open(path, 'r',encoding='utf8') as reader:
            for line in reader:
                if "IMGID" in line: 
                    a=1
                else:
                    line = line.strip()
                    cols = line.split('\t')
                    if len(cols) < 2:
                        if len(tmp_tok) > 0:
                            tokens.append(tmp_tok); labels.append(tmp_lab)
                        tmp_tok = []
                        tmp_lab = []
                    else:
                        tmp_tok.append(cols[0])
                        tmp_lab.append(cols[-1])
                        label_set.append(cols[-1])
                        
        dict_list = []
        for token_seq, label_seq in zip(tokens, labels):
            text = ' '.join(token_seq)  # Ghép các từ lại thành một chuỗi văn bản
            dict_list.append({'text': text, 'label': label_seq})
    return  dict_list


def decode_bio_tags(tags):
    """decode entity (type, start, end) from BIO style tags
    """
    chunks = []
    chunk = [-1, -1, -1]
    for i, tag in enumerate(tags):

        if tag.startswith('B-'):
            if chunk[2] != -1:
                chunks.append(chunk)

            chunk = [-1, -1, -1]
            chunk[0] = tag.split('-')[1]
            chunk[1] = i
            chunk[2] = i + 1
            if i == len(tags) - 1:
                chunks.append(chunk)

        elif tag.startswith('I-') and chunk[1] != -1:
            t = tag.split('-')[1]
            if t == chunk[0]:
                chunk[2] = i + 1

            if i == len(tags) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]

    return chunks
