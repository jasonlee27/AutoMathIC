
import re, os
import sys
import time
import random
import argparse

from typing import *
from pathlib import Path

from .utils.macros import Macros
from .utils.utils import Utils

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--run', type=str, required=True,
                    choices=[
                        'text2code',
                        'mutate',
                        'mutate_nl',
                        'mutate_nl_with_validation_cls',
                        'evaluate_llm',
                        'evaluate_mm_llm',
                        'evaluate_demo_llm',
                        'evaluate_alg_consist_demo_llm',
                        'evaluate_alg_consist_code_llm',
                        'evaluate_modconst_n_discriminator',
                        'measure_consistency',
                        'metric_llm',
                        'asdiv_q_analysis',
                        'resp_analysis',
                        'dataset_analysis',
                        'gen_code_for_alg_consistency',
                        'suspiciousness_exp',
                        'suspiciousness_analysis',
                        'alg_consistency_analysis',
                        'generate_neg_examples_for_discriminator',
                        'evaluate_for_genetic_alg',
                        'genetic_alg',
                        'genetic_fg_alg',
                        'genetic_fg_w_val_cls_alg',
                        'genetic_fg2_alg',
                        'genetic_alg_w_selfconsistency',
                        'metric_genetic_alg',
                        'metric_genetic_fg_alg',
                        'metric_genetic_fg_alg_w_val_cls',
                        'metric_genetic_alg_w_selfconsistency',
                        'sample_data_for_anlyze_mut_correctness',
                        'autocot',
                        'autocot_analysis',
                        'metric_autocot',
                        'logprob_analysis',
                        'generate_csv_for_googlesheet',
                        'genetic_fg_mut_correctness_analysis',
                        'ablation_study'
                    ], help='task to be run')
parser.add_argument('--llm4pl_name', type=str, default='codet5p',
                    choices=[
                        'codet5p', 'text-davinci-002', 'palm',
                    ], help='model name for llm4pl for program synthesis')
parser.add_argument('--dataset_name', type=str, default='asdiv',
                    choices=Macros.ari_rs_dataset_names, help='dataset under test')
parser.add_argument('--llm_model_name', type=str, default='palm',
                    help='name of llm model under test')
parser.add_argument('--include_llm_explanation', action='store_true',
                    help='Indicator if llm response include its explanation for the answer or not')

args = parser.parse_args()


def run_text2code():
    st = time.time()
    from .model.codegen import Codegen_llm, Codegen_eq
    dataset_name = args.dataset_name
    # model_name = args.llm4pl_name
    # Codegen_llm.main(
    #     model_name=model_name, 
    #     dataset_name=dataset_name
    # )
    Codegen_eq.main(dataset_name=dataset_name)
    et = time.time()
    print(f"run_text2code: {et-st} sec")
    return

def run_mutate():
    st = time.time()
    from .mutation.mutate import Mutate
    dataset_name = args.dataset_name
    Mutate.mutate(dataset_name=dataset_name)
    et = time.time()
    print(f"run_mutate: {et-st} sec")
    return

def run_evaluate_llm():
    st = time.time()
    from .llmut.evaluate import Evaluate
    dataset_name = args.dataset_name
    llm_model_name = args.llm_model_name
    include_exp = args.include_llm_explanation
    Evaluate.main(
        model_name=llm_model_name,
        dataset_name=dataset_name,
        include_exp=include_exp
    )
    et = time.time()
    print(f"run_evaluate_llm: {et-st} sec")
    return

def run_evaluate_mm_llm():
    # this method is to get the responses on 
    # code and cot modals over 
    # original and mutated questions.
    st = time.time()
    from .llmut.evaluate import EvaluateWithMultimodals
    dataset_name = args.dataset_name
    llm_model_name = args.llm_model_name
    EvaluateWithMultimodals.main(
        model_name=llm_model_name,
        dataset_name=dataset_name
    )
    et = time.time()
    print(f"run_evaluate_mm_llm::{dataset_name}: {et-st} sec")
    return

def run_evaluate_demo_llm():
    # given the responses on code and cot modals original and mutated questions,
    # it selects the mutated questions and their responses based on the consistency
    # and perform the demonstration learning
    st = time.time()
    from .llmut.evaluate import EvaluateWithDemo
    from .metric.metrics import AccuracyForDemo
    dataset_name = args.dataset_name
    llm_model_name = args.llm_model_name
    # emb_consist_types = list(AccuracyForDemo.emb_consist_type_n_res_file_map.keys())
    emb_consist_types = ['random', 'modcons-random']
    
    for emb_consist_type in emb_consist_types:
        print(f"EMB_CONSIST_TYPE:{emb_consist_type}")
        EvaluateWithDemo.main(
            model_name=llm_model_name,
            dataset_name=dataset_name,
            emb_consist_type=emb_consist_type
        )
    # end if
    et = time.time()
    print(f"run_evaluate_demo_llm ({emb_consist_type}): {et-st} sec")
    return

def run_evaluate_alg_consist_demo_llm():
    st = time.time()
    from .llmut.evaluate import EvaluateWithAlgConsistencyDemo
    dataset_name = args.dataset_name
    llm_model_name = args.llm_model_name

    EvaluateWithAlgConsistencyDemo.main(
        model_name=llm_model_name,
        dataset_name=dataset_name,
        include_only_modconst=False
    )
    EvaluateWithAlgConsistencyDemo.main(
        model_name=llm_model_name,
        dataset_name=dataset_name,
        include_only_modconst=True
    )
    et = time.time()
    print(f"run_evaluate_alg_consist_demo_llm: {et-st} sec")
    return

def run_evaluate_alg_consist_code_llm():
    st = time.time()
    from .llmut.evaluate import EvaluateWithAlgConsistCode
    dataset_name = args.dataset_name
    llm_model_name = args.llm_model_name

    EvaluateWithAlgConsistCode.main(
        model_name=llm_model_name,
        dataset_name=dataset_name,
        include_only_modconst=False
    )
    EvaluateWithAlgConsistCode.main(
        model_name=llm_model_name,
        dataset_name=dataset_name,
        include_only_modconst=True
    )
    et = time.time()
    print(f"run_evaluate_alg_consist_code_llm: {et-st} sec")
    return

def run_evaluate_modconst_n_discriminator():
    st = time.time()
    from .llmut.evaluate import EvaluateWithModconstNDiscriminator
    dataset_name = args.dataset_name
    llm_model_name = args.llm_model_name
    
    query_model_name = 'bert-base-uncased'
    code_model_name = 'bert-base-uncased'
    discriminator_out_dim = 256
    discriminaotr_score_threshold = 0.5

    EvaluateWithModconstNDiscriminator.main(
        model_name=llm_model_name,
        dataset_name=dataset_name,
        query_model_name=query_model_name,
        code_model_name=code_model_name,
        discriminator_out_dim=discriminator_out_dim,
        discriminaotr_score_threshold=discriminaotr_score_threshold
    )
    et = time.time()
    print(f"run_evaluate_modconst_n_discriminator: {et-st} sec")
    return

def run_measure_consistency():
    # given the responses on code and cot modals original and mutated questions,
    # it selects the mutated questions and their responses based on the consistency
    # and perform the demonstration learning
    st = time.time()
    from .consistency.consistency import ModalConsistency, EmbeddingConsistency
    dataset_name = args.dataset_name
    llm_model_name = args.llm_model_name
    # modal consistency
    ModalConsistency.get_consistency(
        model_name=llm_model_name,
        dataset_name=dataset_name
    )
    # # embedding consistency
    # EmbeddingConsistency.get_consistency(
    #     model_name=llm_model_name,
    #     dataset_name=dataset_name
    # )
    et = time.time()
    print(f"run_measure_consistency::{dataset_name}: {et-st} sec")
    return

def run_metric_llm():
    st = time.time()
    from .metric.metrics import Accuracy, \
        AccuracyForDemo, \
        AccuracyForDemoFromAlgConsistency, \
        AccuracyForCodeFromAlgConsistency, \
        AccuracyForModconstNDiscriminator
    dataset_name = args.dataset_name
    llm_model_name = args.llm_model_name
    emb_consist_types = ['cos_sim', 'dist', 'avg_dist_among_muts']
    # accuracy
    # for emb_type in emb_consist_types:
    #     print(f"EMB_CONSISTENCY:{emb_type}")
    #     AccuracyForDemo.evaluate(
    #         model_name=llm_model_name,
    #         dataset_name=dataset_name,
    #         emb_consist_type=emb_type
    #     )
    # # end for
    AccuracyForDemo.evaluate(
        model_name=llm_model_name,
        dataset_name=dataset_name,
        emb_consist_type=None
    )
    # Accuracy.evaluate(
    #     model_name=llm_model_name,
    #     dataset_name=dataset_name
    # )
    # AccuracyForDemoFromAlgConsistency.evaluate(
    #     model_name=llm_model_name,
    #     dataset_name=dataset_name,
    #     include_only_modconst=False
    # )
    # AccuracyForDemoFromAlgConsistency.evaluate(
    #     model_name=llm_model_name,
    #     dataset_name=dataset_name,
    #     include_only_modconst=True
    # )
    # AccuracyForCodeFromAlgConsistency.evaluate(
    #     model_name=llm_model_name,
    #     dataset_name=dataset_name,
    #     include_only_modconst=False
    # )
    # AccuracyForCodeFromAlgConsistency.evaluate(
    #     model_name=llm_model_name,
    #     dataset_name=dataset_name,
    #     include_only_modconst=True
    # )
    # AccuracyForModconstNDiscriminator.evaluate(
    #     model_name=llm_model_name,
    #     dataset_name=dataset_name
    # )
    et = time.time()
    print(f"run_metric_llm: {et-st} sec")
    return

def run_mutate_nl():
    st = time.time()
    from .mutation.mutate2nl import Mutate2nl, Mutate2nlWoEquation, Mutate2nlWoEquationNModConsistency
    dataset_name = args.dataset_name
    llm_model_name = args.llm_model_name
    # Mutate2nl.mutate(dataset_name=dataset_name)
    Mutate2nlWoEquation.mutate(dataset_name=dataset_name)
    # Mutate2nlWoEquationNModConsistency.mutate(
    #     dataset_name=dataset_name,
    #     llm_name=llm_model_name
    # )
    et = time.time()
    print(f"run_mutate_nl::{dataset_name}: {et-st} sec")
    return

def run_mutate_nl_with_validation_cls():
    st = time.time()
    from .mutation.mutate2nl import Mutate2nlWoEquationNValidationCls
    dataset_name = args.dataset_name
    llm_model_name = args.llm_model_name
    Mutate2nlWoEquationNValidationCls.mutate(
        dataset_name=dataset_name,
        model_name=llm_model_name
    )
    et = time.time()
    print(f"run_mutate_nl_with_validation_cls::{dataset_name}::{llm_model_name}: {et-st} sec")
    return

def run_asdiv_q_analysis():
    from .dataset.asdiv import Asdiv
    Asdiv.organize_qualitative_analysis()
    return

def run_resp_analysis():
    st = time.time()
    from .analyze.analyzeresponse import AnalyzeResponse, AnalyzeGPTResponse
    llm_model_name = args.llm_model_name
    dataset_name = args.dataset_name
    # AnalyzeResponse.write_trimmed_response(
    #     llm_model_name, 
    #     dataset_name
    # )
    AnalyzeGPTResponse.main(
        llm_model_name,
        dataset_name
    )
    et = time.time()
    print(f"run_resp_analysis: {et-st} sec")
    return

def run_dataset_analysis():
    st = time.time()
    from .analyze.analyzedataset import AnalyzeDataset
    dataset_name = args.dataset_name
    AnalyzeDataset.get_tokens_stat(dataset_name)
    et = time.time()
    print(f"run_dataset_analysis: {et-st} sec")
    return

def run_gen_code_for_alg_consistency():
    st = time.time()
    from .consistency.algconsistency import AlgorithmConsistency
    dataset_name = args.dataset_name
    llm_model_name = args.llm_model_name
    # taking only the modal-consistent mutations in consideration
    AlgorithmConsistency.main(
        dataset_name, 
        llm_model_name,
        include_only_modconst=True
    )
    # taking all mutations in consideration
    AlgorithmConsistency.main(
        dataset_name, 
        llm_model_name,
        include_only_modconst=False
    )
    et = time.time()
    print(f"run_gen_code_for_alg_consistency: {et-st} sec")
    return

def run_suspiciousness_exp():
    st = time.time()
    from .suspiciousness.retrieve import Retrieve
    dataset_name = args.dataset_name
    llm_model_name = args.llm_model_name
    ext_dataset_name = 'asdiv'
    print(f"dataset_under_test_name:{dataset_name}")
    print(f"ext_dataset_name:{ext_dataset_name}")
    Retrieve.exp_main( 
        dataset_name=dataset_name,
        ext_dataset_name=ext_dataset_name,
        model_name=llm_model_name,
        retrival_method='embedding',
        embedding_model_name='openai',
        topk=5
    )
    et = time.time()
    print(f"run_suspiciousness_exp: {et-st} sec")
    return

def run_suspiciousness_analysis():
    st = time.time()
    from .analyze.analyzesuspiciousness import AnalyzeSuspiciousness
    dataset_name = args.dataset_name
    # llm_model_name = args.llm_model_name
    ext_dataset_name = 'asdiv'
    print(f"dataset_under_test_name:{dataset_name}")
    print(f"ext_dataset_name:{ext_dataset_name}")
    AnalyzeSuspiciousness.get_correctness( 
        dataset_name=dataset_name,
        ext_dataset_name=ext_dataset_name,
        retrival_method='embedding'
    )
    et = time.time()
    print(f"run_suspiciousness_analysis: {et-st} sec")
    return

def run_alg_consistency_analysis():
    st = time.time()
    from .analyze.analyzealgorithmconsistency import AnalyzeAlgorithmConsistency
    dataset_name = args.dataset_name
    llm_model_name = args.llm_model_name
    print(f"dataset_under_test_name:{dataset_name}")
    print(f"llm_model_name:{llm_model_name}")
    # AnalyzeAlgorithmConsistency.generate_matrix_csv_over_all_alg_consts( 
    #     dataset_name=dataset_name,
    #     model_name=llm_model_name
    # )
    AnalyzeAlgorithmConsistency.get_overlap_with_baselines( 
        dataset_name=dataset_name,
        model_name=llm_model_name,
        include_only_modconst=False
    )
    AnalyzeAlgorithmConsistency.get_overlap_with_baselines( 
        dataset_name=dataset_name,
        model_name=llm_model_name,
        include_only_modconst=True
    )
    AnalyzeAlgorithmConsistency.get_overlap_btw_with_n_without_modconst( 
        dataset_name=dataset_name,
        model_name=llm_model_name
    )
    et = time.time()
    print(f"run_alg_consistency_analysis: {et-st} sec")
    return

def run_generate_neg_examples_for_discriminator():
    st = time.time()
    from .discriminator.mutate import Mutate
    dataset_name = args.dataset_name

    code_str = "\n\ndef func():\n    return 22 * 46\nthe answer is 1012"
    answer = '1012'
    res_dir = Macros.result_dir / 'discriminator' / dataset_name
    res_dir.mkdir(parents=True, exist_ok=True)
    num_neg_examples = 10
    
    mut_obj = Mutate(
        code_str=code_str,
        answer=answer,
        res_dir=res_dir
    )
    mut_obj.generate_negative_examples(
        num_neg_examples=num_neg_examples
    )
    et = time.time()
    print(f"run_generate_neg_examples_for_discriminator: {et-st} sec")
    return

def run_evaluate_for_genetic_alg():
    st = time.time()
    from .genetic.multimodals import Multimodals
    dataset_name = args.dataset_name
    llm_name = args.llm_model_name
    Multimodals.main(
        dataset_name=dataset_name,
        llm_name=llm_name
    )
    et = time.time()
    print(f"run_evaluate_for_genetic_alg::{dataset_name} {et-st} sec")
    return

def run_genetic_alg():
    st = time.time()
    from .genetic.genetic import Genetic
    dataset_name = args.dataset_name
    llm_name = args.llm_model_name
    Genetic.main(
        dataset_name=dataset_name,
        llm_name=llm_name
    )
    et = time.time()
    print(f"run_genetic_alg: {et-st} sec")
    return

def run_genetic_fg_alg():
    st = time.time()
    from .genetic.genetic_fg import GeneticFineGrained
    dataset_name = args.dataset_name
    llm_name = args.llm_model_name
    GeneticFineGrained.main(
        dataset_name=dataset_name,
        llm_name=llm_name
    )
    et = time.time()
    print(f"run_genetic_fg_alg: {et-st} sec")
    return

def run_genetic_fg_w_val_cls_alg():
    st = time.time()
    from .genetic.genetic_fg import GeneticFineGrainedWithValidationCls
    dataset_name = args.dataset_name
    llm_name = args.llm_model_name
    GeneticFineGrainedWithValidationCls.main(
        dataset_name=dataset_name,
        llm_name=llm_name
    )
    et = time.time()
    print(f"run_genetic_fg_w_val_cls_alg: {et-st} sec")
    return

def run_genetic_fg2_alg():
    st = time.time()
    from .genetic.genetic_fg2 import GeneticFineGrainedWithDynamicMutSelection
    dataset_name = args.dataset_name
    llm_name = args.llm_model_name
    GeneticFineGrainedWithDynamicMutSelection.main(
        dataset_name=dataset_name,
        llm_name=llm_name
    )
    et = time.time()
    print(f"run_genetic_fg2_alg: {et-st} sec")
    return

def run_genetic_alg_w_selfconsistency():
    st = time.time()
    from .genetic.genetic import GeneticUsingSelfConsistency
    dataset_name = args.dataset_name
    llm_name = args.llm_model_name
    print(f"run_genetic_alg_w_selfconsistency::{dataset_name}...")
    GeneticUsingSelfConsistency.main(
        dataset_name=dataset_name,
        llm_name=llm_name,
        target_modality='cot_response',
        top_k=len(Macros.prompts_over_modals.keys()),
        temp=0.7 # Macros.resp_temp
    )
    et = time.time()
    print(f"run_genetic_alg_w_selfconsistency: {et-st} sec")
    return

def run_metric_genetic_alg():
    st = time.time()
    # from .genetic.genetic import Genetic
    from .metric.metrics import AccuracyForEachModal, AccuracyForGeneticAlg
    dataset_name = args.dataset_name
    llm_name = args.llm_model_name
    AccuracyForEachModal.evaluate(
        llm_name=llm_name,
        dataset_name=dataset_name
    )
    AccuracyForGeneticAlg.evaluate(
        llm_name=llm_name,
        dataset_name=dataset_name
    )
    # AccuracyForGeneticAlg.compare_with_baseline(
    #     llm_name=llm_name,
    #     dataset_name=dataset_name
    # )
    et = time.time()
    print(f"run_metric_genetic_alg::{dataset_name}: {et-st} sec")
    return

def run_metric_genetic_fg_alg():
    st = time.time()
    # from .genetic.genetic import Genetic
    from .metric.metrics import AccuracyForGeneticFgAlg, AccuracyForEachModal
    dataset_name = args.dataset_name
    llm_name = args.llm_model_name
    # AccuracyForEachModal.evaluate(
    #     llm_name=llm_name,
    #     dataset_name=dataset_name
    # )
    AccuracyForGeneticFgAlg.evaluate(
        llm_name=llm_name,
        dataset_name=dataset_name
    )
    et = time.time()
    print(f"run_metric_genetic_fg_alg::{dataset_name}: {et-st} sec")
    return

def run_metric_genetic_fg_alg_w_val_cls():
    st = time.time()
    # from .genetic.genetic import Genetic
    from .metric.metrics import AccuracyForGeneticFgAlgWithValidationCls
    dataset_name = args.dataset_name
    llm_name = args.llm_model_name
    AccuracyForGeneticFgAlgWithValidationCls.evaluate(
        llm_name=llm_name,
        dataset_name=dataset_name
    )
    et = time.time()
    print(f"run_metric_genetic_fg_alg_w_val_cls::{dataset_name}: {et-st} sec")
    return

def run_metric_genetic_alg_w_selfconsistency():
    st = time.time()
    # from .genetic.genetic import Genetic
    from .metric.metrics import AccuracyForGeneticAlgSelfConsistency
    dataset_name = args.dataset_name
    llm_name = args.llm_model_name
    AccuracyForGeneticAlgSelfConsistency.evaluate(
        llm_name=llm_name,
        dataset_name=dataset_name,
        target_modality='cot_response'
    )
    et = time.time()
    print(f"run_metric_genetic_alg_w_selfconsistency::{dataset_name}: {et-st} sec")
    return

def run_sample_data_for_anlyze_mut_correctness():
    st = time.time()
    # from .genetic.genetic import Genetic
    from .analyze.analyzemutation import AnalyzeMutation, AnalyzeConsistency
    llm_name = args.llm_model_name
    dataset_names = Macros.ari_rs_dataset_names
    for dataset_name in dataset_names:
        print(f"DATASET::{dataset_name}")
        AnalyzeMutation.get_sample_data(
            llm_name=llm_name,
            dataset_name=dataset_name,
            sample_size=100
        )
        # AnalyzeConsistency.get_consistency_stats(
        #     llm_name=llm_name,
        #     dataset_name=dataset_name
        # )
        # AnalyzeConsistency.get_mutation_consistency(
        #     llm_name=llm_name,
        #     dataset_name=dataset_name
        # )
    # end for
    et = time.time()
    print(f"run_sample_data_for_anlyze_mut_correctness::{dataset_names}: {et-st} sec")
    return

def run_generate_csv_for_googlesheet():
    st = time.time()
    from .analyze.analyzemutation import AnalyzeCSVgenForDataAnalysis
    llm_name = args.llm_model_name
    AnalyzeCSVgenForDataAnalysis.generate_csv_gsm8k(llm_name=llm_name)
    return

def run_autocot():
    st = time.time()
    from .baseline.autocot import AutoCot
    llm_name = args.llm_model_name
    dataset_name = args.dataset_name
    AutoCot.main(
        model_name=llm_name,
        dataset_name=dataset_name
    )
    et = time.time()
    print(f"run_autocot::{dataset_name}: {et-st} sec")
    return

def run_metric_autocot():
    st = time.time()
    from .metric.metrics import AccuracyForAutoCot
    llm_name = args.llm_model_name
    dataset_name = args.dataset_name
    AccuracyForAutoCot.evaluate(
        llm_name=llm_name,
        dataset_name=dataset_name
    )
    et = time.time()
    print(f"run_metric_autocot::{dataset_name}: {et-st} sec")
    return

def run_autocot_analysis():
    st = time.time()
    from .analyze.analyzeautocot import AnalyzeAutocot
    llm_name = args.llm_model_name
    dataset_name = args.dataset_name
    AnalyzeAutocot.main(
        llm_name=llm_name,
        dataset_name=dataset_name
    )
    et = time.time()
    print(f"run_autocot_analysis::{dataset_name}: {et-st} sec")
    return

def run_logprob_analysis():
    st = time.time()
    from .analyze.analyzelogprob import AnalyzeLogprob
    llm_name = args.llm_model_name
    dataset_name = args.dataset_name
    AnalyzeLogprob.main(
        llm_name=llm_name,
        dataset_name=dataset_name
    )
    et = time.time()
    print(f"run_logprob_analysis::{dataset_name}: {et-st} sec")
    return

def run_genetic_fg_mut_correctness_analysis():
    st = time.time()
    from .analyze.analyzemutation import AnalyzeMutationCorrectness
    llm_name = args.llm_model_name
    dataset_name = args.dataset_name
    AnalyzeMutationCorrectness.print_cksum_vals(
        llm_name=llm_name,
        dataset_name=dataset_name
    )
    et = time.time()
    print(f"run_genetic_fg_mut_correctness_analysis::{dataset_name}: {et-st} sec")
    return 

def run_ablation_study():
    st = time.time()
    from .analyze.ablationstudy import AblationStudy
    llm_name = args.llm_model_name
    dataset_name = args.dataset_name
    AblationStudy.main_phaseone_only(
        dataset_name,
        llm_name
    )
    AblationStudy.analyze(
        dataset_name,
        llm_name
    )
    et = time.time()
    print(f"run_ablation_study::{dataset_name}: {et-st} sec")
    return

func_map = {
    'text2code': run_text2code,
    'mutate': run_mutate,
    'mutate_nl': run_mutate_nl,
    'mutate_nl_with_validation_cls': run_mutate_nl_with_validation_cls,
    'evaluate_llm': run_evaluate_llm,
    'evaluate_mm_llm': run_evaluate_mm_llm,
    'evaluate_demo_llm': run_evaluate_demo_llm,
    'evaluate_alg_consist_demo_llm': run_evaluate_alg_consist_demo_llm,
    'evaluate_alg_consist_code_llm': run_evaluate_alg_consist_code_llm,
    'evaluate_modconst_n_discriminator': run_evaluate_modconst_n_discriminator,
    'measure_consistency': run_measure_consistency,
    'metric_llm': run_metric_llm,
    'asdiv_q_analysis': run_asdiv_q_analysis,
    'resp_analysis': run_resp_analysis,
    'dataset_analysis': run_dataset_analysis,
    'gen_code_for_alg_consistency': run_gen_code_for_alg_consistency,
    'suspiciousness_exp': run_suspiciousness_exp,
    'suspiciousness_analysis': run_suspiciousness_analysis,
    'alg_consistency_analysis': run_alg_consistency_analysis,
    'generate_neg_examples_for_discriminator': run_generate_neg_examples_for_discriminator,
    'evaluate_for_genetic_alg': run_evaluate_for_genetic_alg,
    'genetic_alg': run_genetic_alg,
    'genetic_fg_alg': run_genetic_fg_alg,
    'genetic_fg_w_val_cls_alg': run_genetic_fg_w_val_cls_alg,
    'metric_genetic_fg_alg_w_val_cls': run_metric_genetic_fg_alg_w_val_cls,
    'genetic_fg2_alg': run_genetic_fg2_alg,
    'genetic_alg_w_selfconsistency': run_genetic_alg_w_selfconsistency,
    'metric_genetic_alg': run_metric_genetic_alg,
    'metric_genetic_fg_alg': run_metric_genetic_fg_alg,
    'metric_genetic_alg_w_selfconsistency': run_metric_genetic_alg_w_selfconsistency,
    'sample_data_for_anlyze_mut_correctness': run_sample_data_for_anlyze_mut_correctness,
    'autocot': run_autocot,
    'autocot_analysis': run_autocot_analysis,
    'metric_autocot': run_metric_autocot,
    'logprob_analysis': run_logprob_analysis,
    'generate_csv_for_googlesheet': run_generate_csv_for_googlesheet,
    'genetic_fg_mut_correctness_analysis': run_genetic_fg_mut_correctness_analysis,
    'ablation_study': run_ablation_study
}

if __name__=="__main__":
    func_map[args.run]()
