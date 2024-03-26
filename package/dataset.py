# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset, DataLoader
from package.utils import read_clue_json

# tag2id = {'O': 0,
#           'B-address': 1, 'I-address': 2,
#           'B-book': 3, 'I-book': 4,
#           'B-company': 5, 'I-company': 6,
#           'B-game': 7, 'I-game': 8,
#           'B-government': 9, 'I-government': 10,
#           'B-movie': 11, 'I-movie': 12,
#           'B-name': 13, 'I-name': 14,
#           'B-organization': 15, 'I-organization': 16,
#           'B-position': 17, 'I-position': 18,
#           'B-scene': 19, 'I-scene': 20}

# id2tag = {0: 'O',
#           1: 'B-address', 2: 'I-address',
#           3: 'B-book', 4: 'I-book',
#           5: 'B-company', 6: 'I-company',
#           7: 'B-game', 8: 'I-game',
#           9: 'B-government', 10: 'I-government',
#           11: 'B-movie', 12: 'I-movie',
#           13: 'B-name', 14: 'I-name',
#           15: 'B-organization', 16: 'I-organization',
#           17: 'B-position', 18: 'I-position',
#           19: 'B-scene', 20: 'I-scene'}

tag2id={'O': 0,
 'B-LOCATION-GPE': 1,
 'I-LOCATION-GPE': 2,
 'B-QUANTITY-NUM': 3,
 'B-EVENT-CUL': 4,
 'I-EVENT-CUL': 5,
 'B-DATETIME': 6,
 'I-DATETIME': 7,
 'B-DATETIME-DATERANGE': 8,
 'I-DATETIME-DATERANGE': 9,
 'B-PERSONTYPE': 10,
 'B-PERSON': 11,
 'B-QUANTITY-PER': 12,
 'I-QUANTITY-PER': 13,
 'B-ORGANIZATION': 14,
 'B-LOCATION-GEO': 15,
 'I-LOCATION-GEO': 16,
 'B-LOCATION-STRUC': 17,
 'I-LOCATION-STRUC': 18,
 'B-PRODUCT-COM': 19,
 'I-PRODUCT-COM': 20,
 'I-ORGANIZATION': 21,
 'B-DATETIME-DATE': 22,
 'I-DATETIME-DATE': 23,
 'B-QUANTITY-DIM': 24,
 'I-QUANTITY-DIM': 25,
 'B-PRODUCT': 26,
 'I-PRODUCT': 27,
 'B-QUANTITY': 28,
 'I-QUANTITY': 29,
 'B-DATETIME-DURATION': 30,
 'I-DATETIME-DURATION': 31,
 'I-PERSON': 32,
 'B-QUANTITY-CUR': 33,
 'I-QUANTITY-CUR': 34,
 'B-DATETIME-TIME': 35,
 'B-QUANTITY-TEM': 36,
 'I-QUANTITY-TEM': 37,
 'B-DATETIME-TIMERANGE': 38,
 'I-DATETIME-TIMERANGE': 39,
 'B-EVENT-GAMESHOW': 40,
 'I-EVENT-GAMESHOW': 41,
 'B-QUANTITY-AGE': 42,
 'I-QUANTITY-AGE': 43,
 'B-QUANTITY-ORD': 44,
 'I-QUANTITY-ORD': 45,
 'B-PRODUCT-LEGAL': 46,
 'I-PRODUCT-LEGAL': 47,
 'I-PERSONTYPE': 48,
 'I-DATETIME-TIME': 49,
 'B-LOCATION': 50,
 'B-ORGANIZATION-MED': 51,
 'I-ORGANIZATION-MED': 52,
 'B-URL': 53,
 'B-PHONENUMBER': 54,
 'B-ORGANIZATION-SPORTS': 55,
 'I-ORGANIZATION-SPORTS': 56,
 'B-EVENT-SPORT': 57,
 'I-EVENT-SPORT': 58,
 'B-SKILL': 59,
 'I-SKILL': 60,
 'B-EVENT-NATURAL': 61,
 'I-LOCATION': 62,
 'I-EVENT-NATURAL': 63,
 'I-QUANTITY-NUM': 64,
 'B-EVENT': 65,
 'I-EVENT': 66,
 'B-ADDRESS': 67,
 'I-ADDRESS': 68,
 'B-IP': 69,
 'I-IP': 70,
 'I-PHONENUMBER': 71,
 'B-EMAIL': 72,
 'I-EMAIL': 73,
 'I-URL': 74,
 'B-ORGANIZATION-STOCK': 75,
 'B-DATETIME-SET': 76,
 'I-DATETIME-SET': 77,
 'B-PRODUCT-AWARD': 78,
 'I-PRODUCT-AWARD': 79,
 'B-MISCELLANEOUS': 80,
 'I-MISCELLANEOUS': 81,
 'I-ORGANIZATION-STOCK': 82,
 'B-LOCATION-GPE-GEO': 83}
id2tag={'O': 0,
 'B-LOCATION-GPE': 1,
 'I-LOCATION-GPE': 2,
 'B-QUANTITY-NUM': 3,
 'B-EVENT-CUL': 4,
 'I-EVENT-CUL': 5,
 'B-DATETIME': 6,
 'I-DATETIME': 7,
 'B-DATETIME-DATERANGE': 8,
 'I-DATETIME-DATERANGE': 9,
 'B-PERSONTYPE': 10,
 'B-PERSON': 11,
 'B-QUANTITY-PER': 12,
 'I-QUANTITY-PER': 13,
 'B-ORGANIZATION': 14,
 'B-LOCATION-GEO': 15,
 'I-LOCATION-GEO': 16,
 'B-LOCATION-STRUC': 17,
 'I-LOCATION-STRUC': 18,
 'B-PRODUCT-COM': 19,
 'I-PRODUCT-COM': 20,
 'I-ORGANIZATION': 21,
 'B-DATETIME-DATE': 22,
 'I-DATETIME-DATE': 23,
 'B-QUANTITY-DIM': 24,
 'I-QUANTITY-DIM': 25,
 'B-PRODUCT': 26,
 'I-PRODUCT': 27,
 'B-QUANTITY': 28,
 'I-QUANTITY': 29,
 'B-DATETIME-DURATION': 30,
 'I-DATETIME-DURATION': 31,
 'I-PERSON': 32,
 'B-QUANTITY-CUR': 33,
 'I-QUANTITY-CUR': 34,
 'B-DATETIME-TIME': 35,
 'B-QUANTITY-TEM': 36,
 'I-QUANTITY-TEM': 37,
 'B-DATETIME-TIMERANGE': 38,
 'I-DATETIME-TIMERANGE': 39,
 'B-EVENT-GAMESHOW': 40,
 'I-EVENT-GAMESHOW': 41,
 'B-QUANTITY-AGE': 42,
 'I-QUANTITY-AGE': 43,
 'B-QUANTITY-ORD': 44,
 'I-QUANTITY-ORD': 45,
 'B-PRODUCT-LEGAL': 46,
 'I-PRODUCT-LEGAL': 47,
 'I-PERSONTYPE': 48,
 'I-DATETIME-TIME': 49,
 'B-LOCATION': 50,
 'B-ORGANIZATION-MED': 51,
 'I-ORGANIZATION-MED': 52,
 'B-URL': 53,
 'B-PHONENUMBER': 54,
 'B-ORGANIZATION-SPORTS': 55,
 'I-ORGANIZATION-SPORTS': 56,
 'B-EVENT-SPORT': 57,
 'I-EVENT-SPORT': 58,
 'B-SKILL': 59,
 'I-SKILL': 60,
 'B-EVENT-NATURAL': 61,
 'I-LOCATION': 62,
 'I-EVENT-NATURAL': 63,
 'I-QUANTITY-NUM': 64,
 'B-EVENT': 65,
 'I-EVENT': 66,
 'B-ADDRESS': 67,
 'I-ADDRESS': 68,
 'B-IP': 69,
 'I-IP': 70,
 'I-PHONENUMBER': 71,
 'B-EMAIL': 72,
 'I-EMAIL': 73,
 'I-URL': 74,
 'B-ORGANIZATION-STOCK': 75,
 'B-DATETIME-SET': 76,
 'I-DATETIME-SET': 77,
 'B-PRODUCT-AWARD': 78,
 'I-PRODUCT-AWARD': 79,
 'B-MISCELLANEOUS': 80,
 'I-MISCELLANEOUS': 81,
 'I-ORGANIZATION-STOCK': 82,
 'B-LOCATION-GPE-GEO': 83}


def decode_tags_from_ids(batch_ids):
    batch_tags = []
    for ids in batch_ids:
        sequence_tags = []
        for id in ids:
            sequence_tags.append(id2tag[int(id)])
        batch_tags.append(sequence_tags)
    return batch_tags


class CLUEDataset(Dataset):
    """Pytorch Dataset for CLUE
    """

    def __init__(self, path_to_clue, tokenizer):
        self.data = read_clue_json(path_to_clue)
        self.tokenizer = tokenizer

    def collate_fn(self, batch):
        """collate_fn for 'torch.utils.data.DataLoader'
        """
        texts, labels = list(zip(*[[item[0], item[1]] for item in batch]))
        token = self.tokenizer(list(texts), padding=False, return_offsets_mapping=True)

        # align the label
        # Bert mat split a word 'AA' into 'A' and '##A'
        labels = [self._align_label(offset, label) for offset, label in zip(token['offset_mapping'], labels)]
        token = self.tokenizer.pad(token, padding=True, return_attention_mask=True)

        return torch.LongTensor(token['input_ids']), torch.ByteTensor(token['attention_mask']), self._pad(labels)

    @staticmethod
    def _align_label(offset, label):

        label_align = []
        for i, (start, end) in enumerate(offset):

            if start == end:
                label_align.append(tag2id['O'])
            else:
                # 1-N or N-1, default to use first original label as final label
                if i > 0 and offset[i - 1] == (start, end):
                    label_align.append(label[start:end][0].replace('B', 'O', 1))
                else:
                    label_align.append(label[start:end][0])
        return label_align

    @staticmethod
    def _pad(labels):
        max_len = max([len(label) for label in labels])
        labels = [(label + [tag2id['O']] * (max_len - len(label))) for label in labels]
        return torch.LongTensor(labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        pkg = self.data[index]

        text = pkg['text']
        label = [tag2id[tag] for tag in pkg["label"]]

        return text, label
