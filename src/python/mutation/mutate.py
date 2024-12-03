
import os
import re
import random
import subprocess

from typing import *
from pathlib import Path

from ..utils.macros import Macros
from ..utils.utils import Utils
from ..utils.logger import Logger


class Mutate:

    def __init__(
        self, 
        cksum_val: str,
        dataset_name: str,
        pl_type='python',
        seed_num=Macros.RAND_SEED
    ):
        random.seed(seed_num)
        self.cksum_val: str = cksum_val
        self.pl_type = pl_type
        self.res_dir = Macros.result_dir / 'eq2code' / dataset_name / pl_type
        self.code_file: Path = self.res_dir / f"{self.cksum_val}.py"
        self.var_dict_file: Path = self.res_dir / f"vars-{self.cksum_val}.json"
        with open(self.code_file, 'r') as f:
            self.code: str = f.read()
        # end with
        self.var_dict: Dict = Utils.read_json(self.var_dict_file)
        self.var_types: Dict = self.get_var_type(self.var_dict)
        # self.answer = self.var_dict['<answer>']
        # self.answer_type = self.var_types['<answer>']
        self.mut_vals = self.mutate_variable()
        self.mut_codes = self.mutate_code()
        
    def get_var_type(
        self, 
        var_dict: Dict
    ) -> Dict:
        return {
            var: str(type(val).__name__)
            for var, val in var_dict.items()
        }

    def mutate_by_random_over_vars(
        self,
        var_dict: Dict,
        var_type_dict: Dict,
        num_mutations: int,
        max_bound: Any,
        min_bound: Any
    ) -> List[Dict]:
        samples = list()
        for m_i in range(num_mutations):
            mut_dict = dict()
            for var in var_dict.keys():
                if var!=Macros.answer_key_for_var_dict:
                    orig_val = var_dict[var]
                    val_type = var_type_dict[var]
                    mut_val = self.mutate_by_random_per_var(
                        orig_val,
                        val_type,
                        max_bound,
                        min_bound
                    )
                    if mut_val is not None:
                        mut_dict[var] = mut_val
                    # end if
                # end if
            # end for
            if mut_dict not in samples:
                samples.append(mut_dict)
            # end if
        # end for
        return samples
    
    def mutate_by_random_per_var(
        self,
        value: Any,
        value_type: str,
        max_bound: Any,
        min_bound: Any
    ) -> Any:
        # in this method, we select values randomly 
        # but same type of original variable value
        val = None
        if value_type=='int':
            # TODO: random select int over the num_mutation
            val = random.randint(min_bound, max_bound)
        elif value_type=='float':
            # TODO: random select float over the num_mutation
            whole = random.randint(min_bound, max_bound-1)
            fraction = random.random() # sample random number between 0 and 1
            val = whole*1.+fraction
        elif value_type=='bool':
            # TODO: random select bool over the num_mutation
            val = random.choice([True, False])
        # end if
        return val

    def mutate_variable(
        self, 
        mut_method: str = 'random',
        num_mutations: int = Macros.num_mutations,
        max_bound: int = 100,
        min_bound: int = 0
    ) -> List[Dict]:
        mut_vals = None
        if mut_method=='random':
            mut_vals = self.mutate_by_random_over_vars(
                self.var_dict,
                self.var_types,
                num_mutations,
                max_bound,
                min_bound
            )
        # end if        
        return mut_vals

    def mutate_code(self) -> List[str]:
        mut_codes = list()
        for mut_dict in self.mut_vals:
            mut_code = self.code
            for var in mut_dict.keys():
                pat = var + r'\s=.*'
                new_statement = f"{var} = {mut_dict[var]}"
                mut_code = re.sub(pat, new_statement, mut_code)
            # end for
            mut_codes.append(mut_code)
        # end for
        return mut_codes

    def execute_mut_codes(self, mut_code: str) -> str:
        cmd = None
        mut_dir = self.res_dir / 'mutation'
        mut_dir.mkdir(parents=True, exist_ok=True)

        # write code temporaly
        mut_temp_file_for_exec = mut_dir / 'mut_temp.py'
        Utils.write_txt(mut_code, mut_temp_file_for_exec)

        # execute the mut_temp.py
        if self.pl_type=='python':
            cmd = f"python {str(mut_temp_file_for_exec)}"
        # end if
        output = subprocess.check_output(cmd, shell=True).strip()
        return output.decode()

    def write_code(self) -> None:
        mut_res = list()
        res_dir = self.res_dir / 'mutation'
        res_dir.mkdir(parents=True, exist_ok=True)
        for m_i, mut_code in enumerate(self.mut_codes):
            mut_var_dict = self.mut_vals[m_i]
            mut_out = self.execute_mut_codes(mut_code)
            mut_res.append({
                'code': mut_code,
                'var': mut_var_dict,
                'answer': mut_out
            })
        # end for
        if any(mut_res):
            Utils.write_json(
                mut_res, 
                res_dir / f"mut-{self.cksum_val}.json"
            )
        # end if
        os.remove(str(res_dir / 'mut_temp.py'))
        return
    
    @classmethod
    def mutate(
        cls,
        dataset_name: str,
        pl_type: str='python',
        seed_num: str=Macros.RAND_SEED
    ) -> None:
        res_dir = Macros.result_dir / 'eq2code' / dataset_name / pl_type
        all_orig_codes = [
            f for f in os.listdir(str(res_dir))
            if f.endswith('.py')
        ]
        for orig_code in all_orig_codes:
            cksum_val = orig_code.split('.py')[0]
            mut_obj = cls(
                cksum_val=cksum_val,
                dataset_name=dataset_name,
                pl_type=pl_type,
                seed_num=seed_num
            )
            mut_obj.write_code()
        # end for
        return
