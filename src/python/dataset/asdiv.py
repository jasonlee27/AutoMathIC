

import os
import csv
import numpy as np
# import pandas as pd
import xml.etree.ElementTree as ET

from typing import *
from pathlib import Path
# from datasets import load_dataset
from torch.utils.data import Dataset

from ..utils.macros import Macros
from ..utils.utils import Utils
from ..utils.logger import Logger


class Asdiv(Dataset):
    
    split_fnames = {
        'val': 'ASDiv.xml'
    }
    folds = [
        'fold0.txt', 
        'fold1.txt', 
        'fold2.txt', 
        'fold3.txt', 
        'fold4.txt'
    ]
    
    def __init__(self,
                 split='val') -> None:
        if split not in self.split_fnames.keys():
            valid_splits = '\t'.join(self.split_fnames.keys())
            raise ValueError('Unrecognized split; valid splits are %s' % valid_splits)
        # end if
        self.split = split
        self.root_dir: Path = Macros.dataset_dir / 'asdiv'
        
        if os.path.exists(str(self.root_dir / 'ASDiv.json')):
            raw_data = Utils.read_json(self.root_dir / 'ASDiv.json')
        else:
            raw_data: List[Dict] = self.parse_xml_to_json(self.root_dir / self.split_fnames[self.split])
        # end if

        ids = list()
        for f in self.folds:
            lines = [l.strip() for l in Utils.read_txt(self.root_dir / f)]
            ids.extend(lines)
        # end for
        ids = sorted(list(set(ids)))

        self.id = [r['id'] for r in raw_data if r['id'] in ids]
        self.body = [r['body'] for r in raw_data if r['id'] in ids]
        self.question = [r['question'] for r in raw_data if r['id'] in ids]
        self.type = [r['solution_type'] for r in raw_data if r['id'] in ids]
        self.answer = [r['answer'] for r in raw_data if r['id'] in ids]
        self.equation = [r['equation'] for r in raw_data if r['id'] in ids]
        
    def __len__(self):
        return len(self.id)
    
    def __getitem__(self, index: int):
        return {
            'id': self.id[index],
            'body': self.body[index],
            'question': self.question[index],
            'type': self.type[index],
            'answer': self.answer[index],
            'equation': self.equation[index]
        }


    def parse_xml_to_json(self, xml_file):
        tree = ET.parse(xml_file)
        # get root element
        root = tree.getroot()
        probset = root.find('ProblemSet')

        data = list()
        for prob in probset.findall('Problem'):
            _id = prob.attrib['ID']
            body = prob.find('Body').text
            question = prob.find('Question').text
            solution_type = prob.find('Solution-Type').text
            answer = prob.find('Answer').text
            equation = prob.find('Formula').text
            data.append({
                'id': _id,
                'body': body,
                'question': question,
                'solution_type': solution_type,
                'answer': answer,
                'equation': equation
            })
        # end for
        Utils.write_json(
            data, 
            self.root_dir / 'ASDiv.json', 
            pretty_format=True
        )
        return data
    
    @classmethod
    def organize_qualitative_analysis(cls):
        analysis_file = Macros.root_dir / 'asdiv_qualitative_analysis.csv'
        asdiv_file = Macros.dataset_dir / 'asdiv' / 'ASDiv.json'
        analysis_res, analysis_headers = Utils.read_csv(analysis_file, split_headers=True)
        asdiv_res = Utils.read_json(asdiv_file)
        save_to = Macros.root_dir / 'asdiv_qualitative_analysis_ext.csv'

        with open(save_to, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            headers = analysis_headers + ['body', 'question', 'equation', 'answer']
            csvwriter.writerow(headers)

            for r in analysis_res:
                _id, label, explain = r[0], r[1], r[2]
                for d in asdiv_res:
                    if d['id'].strip()==_id.strip():
                        csvwriter.writerow([
                            _id, label, explain, d['body'], d['question'], d['equation'], d['answer']
                        ])
                    # end if
                # end for
            # end for
        # end with
        return
