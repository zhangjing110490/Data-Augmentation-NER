# -*- coding: utf-8 -*-
# @Time    : 12/22/2022
# @Author  : Jing Zhang

import random
import numpy as np
import json
import os
import argparse


class DataAugmentor:
    def __init__(self, syn_file, eng_chn_file, adj_file, para_file, seed=123):
        random.seed(seed)
        np.random.seed(seed)
        # revise according to data file
        self.entities = 'entities'
        self.text = 'text'
        self.start = 'start_idx'
        self.end = 'end_idx'
        self.type = 'type'
        self.entity = 'entity'

        # load knowledge from file
        self.syn_dict = self.load_synonyms(syn_file)
        self.chn_dict, self.eng_dict = self.load_Eng_CHN(eng_chn_file)
        self.adj_dict = self.load_adj(adj_file)
        self.para_dict = self.load_para(para_file)

    # load synonyms from file
    def load_synonyms(self, data_file, sep=' '):
        with open(data_file, 'r', encoding='utf-8') as fr:
            lines = fr.readlines()
            syn_dict = {}
            for line in lines:
                words = line.strip().split(sep)
                syn_dict[words[0]] = words[1:]
            return syn_dict

    # load English and Chinese for diseases
    def load_Eng_CHN(self, data_file, sep=' || '):
        with open(data_file, 'r', encoding='utf-8') as fr:
            lines = fr.readlines()
            eng_dict, chn_dict = {}, {}
            for line in lines:
                words = line.strip().split(sep)
                eng_dict[words[-1]] = words[0]
                chn_dict[words[0]] = words[-1]
            return chn_dict, eng_dict

    # load adjectives and adverbs for diseases and symptoms
    def load_adj(self, data_file, sep=' '):
        with open(data_file, 'r', encoding='utf-8') as fr:
            lines = fr.readlines()
            adj_dict = {}
            for line in lines:
                words = line.strip().split(sep)
                for word in words:
                    tmp = words.copy()
                    tmp.remove(word)
                    adj_dict[word] = tmp
            return adj_dict

    # load paraphrase for medical entities
    def load_para(self, data_file, sep=' '):
        para_dict = {}
        with open(data_file, 'r', encoding='utf-8') as fr:
            data = fr.readlines()
            for line in data:
                if line.strip() == '':
                    continue
                ents = line.strip().split(sep)
                if len(ents) < 3:
                    continue
                label, ent, para = ents[0], ents[1], ents[2]
                if label not in para_dict:
                    para_dict[label] = {ent: para}
                else:
                    para_dict[label][ent] = para
        return para_dict

    # find synonym for one medical entity
    def SER(self, word):
        if word not in self.syn_dict:
            print(f'synonyms for {word} are not found...')
            return None
        syns = self.syn_dict[word]
        target = random.choice(syns)
        print(f'{target} is chosen as one synonym for {word}...')
        return target

    # find English/Chinese for one medical entity
    def Lan(self, word):
        if word not in self.eng_dict:
            print(f'English for {word} is not found...')
            return None
        eng = self.eng_dict[word]
        print(f'{eng} is chosen as English for {word}...')
        return eng

    # find synonym for one adjective
    def Adj(self, word):
        adjs = list(self.adj_dict.keys())
        if not any([adj in word for adj in adjs]):
            print(f'adj substitute for {word} is not found!')
            return None

        for adj in adjs:
            if adj in word:
                subs = self.adj_dict[adj]
                sub = random.choice(subs)
                start = str(word).index(adj)
                new_word = word[0:start] + sub + word[start + len(adj):]
                print(f'adj substitute {new_word} for {word} is formed!')
                return new_word

    # find paraphase for one medical entity
    def Para(self, sen_index, word):
        if sen_index not in self.para_dict:
            print(f'paraphase for {word} is not found!')
            return None

        if word not in self.para_dict[sen_index]:
            print(f'paraphase for {word} is not found!')
            return None

        para = self.para_dict[sen_index][word]
        print(f'paraphase: {para} for {word} is found!')
        return para

    # find segments [(start, end)...] without medical entities inside
    def find_idle_segments(self, entities, sen_len):
        # return idle segments, in tuple, both start and end included.
        segments = [(ent[self.start], ent[self.end]) for ent in entities]
        segments.sort(key=lambda x: (x[0], x[1]))

        idles = []
        pre = 0
        for start, end in segments:
            if pre <= start - 1:
                idles.append((pre, start - 1))
            pre = max(end+1, pre)
        if pre <= sen_len - 1:
            idles.append((pre, sen_len-1))
        if len(idles) == 0:
            print('None idle segments found for word deletion.')
            print('check the text and entities')

        return idles

    # randomly delete one word (which is not included in any entities) from the input text
    def WD(self, text, entities):
        idles = self.find_idle_segments(entities, len(text))
        if len(idles) == 0:
            print('No WS implemented in this case.')
            return text

        start_, end_ = random.choice(idles)
        target_pos = random.choice(list(range(start_, end_ + 1)))
        print(f'position at {target_pos}: {text[target_pos]} is chosen to be removed.')

        new_text = text[0:target_pos] + text[target_pos + 1:]
        for ent in entities:
            if ent[self.start] > target_pos:
                ent[self.start] -= 1
                ent[self.end] -= 1

        return new_text

    # randomly swap two words (which are not included in any entities) from the input text
    def WS(self, text, entities):
        idles = self.find_idle_segments(entities, len(text))
        idles = [seg for seg in idles if seg[0] < seg[1]]
        if len(idles) == 0:
            print('No WS implemented in this case.')
            return text

        start_, end_ = random.choice(idles)
        target_pos = random.sample(list(range(start_, end_ + 1)), 2)

        i, j = min(target_pos[0], target_pos[1]), max(target_pos[0], target_pos[1])
        print(f'position at {i}: {text[i]}, {j}: {text[j]} are chosen to be removed.')

        new_text = text[0:i] + text[j] + text[i+1:j] + text[i] + text[j+1:]
        return new_text

    # augment input sample{text:, entities:}
    def augment(self, sen_index, sample, operators=['ser', 'lan', 'adj', 'para', 'ws', 'wd']):
        ori_text = sample[self.text]
        new_text = ori_text[:]
        entities = sample[self.entities]
        entities.sort(key=lambda x: (x[self.start], x[self.end]))
        new_entities = []
        augmented = False
        cur_end = -1
        for idx, entity in enumerate(entities):
            if entity[self.start] == -1 and entity[self.end] == -1:
                continue

            if idx > 0 and entity[self.start] <= cur_end:
                if new_text[entity[self.start]:(entity[self.end] + 1)] == entity[self.entity]:
                    new_entities.append(entity)
                    cur_end = max(cur_end, entity[self.end])
                continue

            random.shuffle(operators)
            for oper in operators:
                if oper == 'ser':
                    sub = self.SER(entity[self.entity])
                    if sub is not None:
                        break
                if oper == 'lan':
                    sub = self.Lan(entity[self.entity])
                    if sub is not None:
                        break
                if oper == 'adj':
                    sub = self.Adj(entity[self.entity])
                    if sub is not None:
                        break
                if oper == 'para':
                    if entity[self.type] != 'sym':
                        sub = None
                        continue
                    sub = self.Para(sen_index, entity[self.entity])
                    if sub is not None:
                        break
            if sub is None:
                print('no operation is implemented')
                new_entities.append(entity)
                cur_end = max(cur_end, entity[self.end])
            else:
                augmented = True
                new_entity = (
                    {
                        self.start: entity[self.start],
                        self.end: entity[self.start] + len(sub) - 1,
                        self.type: entity[self.type],
                        self.entity: sub
                    }
                )
                cur_end = max(cur_end, new_entity[self.end])
                new_entities.append(new_entity)
                # update text
                new_text = new_text[0:entity[self.start]] + sub + new_text[entity[self.end] + 1:]
                # update start & end position for subsequent entities
                diff = len(sub) - len(entity[self.entity])
                for j in range(idx + 1, len(entities)):
                    if entities[j][self.start] > entity[self.end]:
                        entities[j][self.start] += diff
                        entities[j][self.end] += diff
                    # update following entities with intersection
                    if entities[j][self.start] <= entity[self.end]:
                        rel_end = entities[j][self.end] - entity[self.end]
                        start_tmp = entities[j][self.start]
                        end_tmp = new_entity[self.end] + rel_end
                        if start_tmp > end_tmp or start_tmp > new_entity[self.end]:
                            entities[j][self.start] = -1
                            entities[j][self.end] = -1
                        else:
                            entities[j][self.end] = end_tmp
                            entities[j][self.entity] = new_text[start_tmp: (end_tmp+1)]

        if not augmented:
            return None
        # if any augment is implemented, add ws and wd
        if 'wd' in operators:
            new_text = self.WD(new_text, new_entities)
        if 'ws' in operators:
            new_text = self.WS(new_text, new_entities)

        new_sample = {
            self.text: new_text,
            self.entities: new_entities
        }
        return new_sample

    @staticmethod
    def save_json_data(data, save_path):
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))

        with open(save_path, 'w', encoding='utf-8') as fw:
            fw.write(json.dumps(data, indent=2, ensure_ascii=False))

    # check the validity of the generated samples
    def check_generated_samples(self, samples):
        flag = True
        for sample in samples:
            text = sample[self.text]
            if '除双肾的对象经血透不能控制的恶性高血压、慢性肾盂肾炎或多囊肾合并感染不易控制时建议移植' in text:
                print('')
            sen_len = len(text)
            ents = sample[self.entities]
            for ent in ents:
                if text[ent[self.start]:(ent[self.end]+1)] != ent[self.entity]:
                    print(f'Generated case with text: {text} is problematic.')
                    flag = False
                if ent[self.start] > ent[self.end]:
                    print(f'start is larger than end.')
                    flag = False
                if ent[self.end] > sen_len - 1:
                    print('end is longer than sentence')
                    flag = False
        return flag

    # augment samples from one data file
    def augment_data_file(self, data_file, save_path, operators, combine=False):
        new_samples = []
        with open(data_file, 'r', encoding='utf-8') as fr:
            samples = json.load(fr)
            for idx, sample in enumerate(samples):
                new_sample = self.augment(idx, sample, operators)
                if new_sample is not None:
                    new_samples.append(new_sample)

        if not self.check_generated_samples(new_samples):
            raise Exception(f'Generation failed with some bad cases.')
        print('All Done')
        print(f'{len(new_samples)} samples are found')
        if combine:
            with open(data_file, 'r', encoding='utf-8') as fr:
                ori_samples = json.load(fr)
            self.save_json_data(new_samples+ori_samples, save_path)
        else:
            self.save_json_data(new_samples, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=123, help='seed for random operations like delete, swap...')
    parser.add_argument('--syn_file', type=str, required=True,
                        help='data file for synonyms of medical entities')
    parser.add_argument('--eng_chn_file', type=str, required=True,
                        help='data file for English and Chinese of diseases')
    parser.add_argument('--adj_file', type=str, required=True,
                        help='data file for adjectives of medical entities')
    parser.add_argument('--para_file', type=str, required=True,
                        help='data file for paraphase of medical entities')
    parser.add_argument('--combine', action='store_true',
                        help='boolean indicating whether to combine added samples to initial samples.')
    parser.add_argument('--data_file', type=str, required=True,
                        help='initial data file to be augmented on.')
    parser.add_argument('--save_path', type=str, required=True,
                        help='data file where the augmented samples are written in.')
    parser.add_argument('--operators', type=str, default="ser adj para ws wd")
    args = parser.parse_args()

    augmentor = DataAugmentor(
        seed=args.seed,
        syn_file=args.syn_file,
        eng_chn_file=args.eng_chn_file,
        adj_file=args.adj_file,
        para_file=args.para_file
    )
    augmentor.augment_data_file(data_file=args.data_file,
                                save_path=args.save_path,
                                operators=str(args.operators).split(' '),
                                combine=args.combine)
