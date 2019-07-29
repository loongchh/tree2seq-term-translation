#!/usr/bin/env python
"""build_data.py - generate raw txt file from downloaded IATE term bank

Written by Riddhiman Dasgupta (https://github.com/dasguptar/treelstm.pytorch)
Rewritten in 2018 by Long-Huei Chen <longhuei@g.ecc.u-tokyo.ac.jp>

To the extent possible under law, the author(s) have dedicated all copyright
and related and neighboring rights to this software to the public domain
worldwide. This software is distributed without any warranty.

You should have received a copy of the CC0 Public Domain Dedication along
with this software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
"""

import sys
sys.path.append(".")
import os
import argparse
import xml.etree.ElementTree as ET
from tqdm import tqdm
from collections import defaultdict

from sklearn.model_selection import train_test_split
from onmt.utils.logging import try_debugger
from tools.test_f1 import normalize_answer


def add_data_opts(parser):
    group = parser.add_argument_group('Data')
    group.add_argument(
        '--tbx-path',
        required=True,
        help="""Where the input TBX file is located in path.""")
    group.add_argument(
        '--save-data',
        help="""Path prefix to which the generated txt files are to be
        saved.""")
    group.add_argument(
        '--save-domain',
        help="""Path prefix to which the generated txt files are to be
        saved.""")
    group.add_argument(
        '--class-path',
        type=str,
        default=
        'lib:lib/stanford-parser/stanford-parser.jar:lib/stanford-parser/stanford-parser-3.9.1-models.jar',
        help="""Class path to be added to Java runtime.""")
    group.add_argument(
        '--test-size',
        type=float,
        default=0.1,
        help='Proportion of train/valid set to use as valid/test set.')
    group.add_argument(
        '--src-lang',
        type=str,
        default="en",
        help="""Source language to be used.""")
    group.add_argument(
        '--tgt-lang',
        type=str,
        default="fr",
        help="""Target language to be used for domain data.""")
    group.add_argument(
        '--lang-list',
        type=str,
        default="""[
            'bg', 'cs', 'da', 'de', 'el', 'en', 'es', 'et', 'fi', 'fr', 'ga',
            'hr', 'hu', 'it', 'la', 'lt', 'lv', 'mt', 'nl', 'pl', 'pt', 'ro',
            'sk', 'sl','sv'
        ]""",
        help="""String list of available languages to be processed.""")
    group.add_argument(
        '--cutoff',
        type=int,
        default=50000,
        help="""Domain size to be included in the dataset.""")
    group.add_argument(
        '--pbar',
        action='store_true',
        help="""Show progress bar.""")


def tbx2dict(src_file, lang_list, src_lang='en', progressbar=False):
    """ Read TermBase eXchange files as Python multiligual dict

    Arguments:
        src_file: (str) path to the TermBase eXchange data file
        lang_list: (str) list of target language to be included
        src_lang: (str) source language of the dict. If the term entry does not
            exist in the source language, the entry is not included

    Output:
        ent_dict: (dict) of multilingual dictionary pointing source language
            terms to another language. ent_dict[lang] is a dictionary of a
            particular language
    """
    tree = ET.parse(src_file)
    root = tree.getroot()
    ent_dict = defaultdict(dict)

    if progressbar:
        pbar = tqdm(total=sum(1 for _ in root[1][0].iter('termEntry')))

    for term in root[1][0].iter('termEntry'):
        src_ent = None
        for term_lang in term.iter('langSet'):
            lang = term_lang.get('{http://www.w3.org/XML/1998/namespace}lang')
            if lang == src_lang:
                src_ent = term_lang[0][0].text

        if src_ent is None or src_ent == "":  # skip if without English entry
            continue

        for term_lang in term.iter('langSet'):
            lang = term_lang.get('{http://www.w3.org/XML/1998/namespace}lang')
            if lang in lang_list and lang != src_lang:
                ent_dict[lang][src_ent] = term_lang[0][0].text

        if progressbar:
            pbar.update(1)

    if progressbar:
        pbar.close()
        s
    return ent_dict


def tbx2domain(src_file, src_lang='en', tgt_lang='fr', progressbar=False):
    """ Read TermBase eXchange files as Python multiligual dict

    Arguments:
        src_file: (str) path to the TermBase eXchange data file
        lang_list: (str) list of target language to be included
        src_lang: (str) source language of the dict. If the term entry does not
            exist in the source language, the entry is not included

    Output:
        ent_dict: (dict) of multilingual dictionary pointing source language
            terms to another language. ent_dict[lang] is a dictionary of a
            particular language
    """
    tree = ET.parse(src_file)
    root = tree.getroot()
    ent_dict = defaultdict(dict)

    if progressbar:
        pbar = tqdm(total=sum(1 for _ in root[1][0].iter('termEntry')))

    for term in root[1][0].iter('termEntry'):
        src_ent = None
        tgt_ent = None
        for term_lang in term.iter('langSet'):
            lang = term_lang.get('{http://www.w3.org/XML/1998/namespace}lang')
            if lang == src_lang:
                src_ent = term_lang[0][0].text
            if lang == tgt_lang:
                tgt_ent = term_lang[0][0].text

        if src_ent is None or src_ent == "":  # skip if without English entry
            continue
        if tgt_ent is None or tgt_ent == "":  # skip if without English entry
            continue

        domain = [int(i[:2]) for i in term.find('descripGrp')[0].text.split(',')]
        for dom in domain:
            ent_dict[dom][src_ent] = tgt_ent

        if progressbar:
            pbar.update(1)

    if progressbar:
        pbar.close()
    return ent_dict


def split_data(src_tgt_dict, maindir, test_size=0.1):
    """ Split dataset into train/valid/test set files.

    Arguments:
        ent_dict: (dict) of multilingual dictionary pointing source language
            terms to another language
        maindir: (str) directory path of the src-tgt language pair
        tgt_lang: (str) target language
        test_size: (float) size of valid/test set relative to train set
    """
    lists = [(normalize_answer(src), normalize_answer(tgt))
             for src, tgt in src_tgt_dict.items()]
    lists = [(s, t) for s, t in lists if s and t]
    src_list, tgt_list = zip(*lists)

    # Split data lists to train/valid/test set
    src_train, src_test, tgt_train, tgt_test = train_test_split(
        src_list, tgt_list, test_size=test_size)
    src_train, src_valid, tgt_train, tgt_valid = train_test_split(
        src_train, tgt_train, test_size=test_size)

    for set_name in ('train', 'valid', 'test'):
        setdir = os.path.join(maindir, set_name)
        if not os.path.isdir(setdir):
            os.makedirs(setdir)

        # Write set as newline-delimited files under separate dirs
        with open(os.path.join(setdir, 'source.txt'), 'w') as f_src:
            f_src.write('\n'.join(eval('src_' + set_name)))
        with open(os.path.join(setdir, 'target.txt'), 'w') as f_tgt:
            f_tgt.write('\n'.join(eval('tgt_' + set_name)))


def dependency_parse(maindir, classpath, tokenize=True):
    """ Dependency parsing done with CoreNLP parser.

    Arguments:
        maindir: (str) directory path of the src-tgt language pair
        cp: (str) class path from which Java is to be called
        tokenize: (bool) whether to tokenize during dependency parse
    """
    for set_name in ('train', 'valid', 'test'):
        dirpath = os.path.join(maindir, set_name)
        filepath = os.path.join(dirpath, 'source.txt')

        filepre = os.path.splitext(os.path.basename(filepath))[0]
        tokpath = os.path.join(dirpath, filepre + '.tok')
        parentpath = os.path.join(dirpath, filepre + '.parent')
        relpath = os.path.join(dirpath, filepre + '.rel')
        tokenize_flag = '-tokenize - ' if tokenize else ''
        cmd = 'java -cp {:s} DependencyParse -tokpath {:s} -parentpath {:s} -relpath {:s} {:s} < {:s}'.format(
            classpath, tokpath, parentpath, relpath, tokenize_flag, filepath)
        os.system(cmd)


def lang_data(src_lang, lang_list, tbx_path, class_path, save_data, test_size=0.1, pbar=False):
    print("Building IATE term dataset by language...")
    ent_dict = tbx2dict(tbx_path, lang_list, src_lang, pbar)
    cmd = "/usr/bin/javac -cp " + class_path + " lib/*.java"
    os.system(cmd)

    with tqdm(total=len(lang_list) - 1) as pbar:
        for tgt_lang in lang_list:  # for each possible target language
            if tgt_lang == src_lang:
                continue
            pbar.set_description(tgt_lang)

            # Split into separate data set files
            maindir = os.path.join(save_data,
                                   src_lang + '_' + tgt_lang)
            split_data(ent_dict[tgt_lang], maindir, test_size=test_size)

            # Dependency parsing
            dependency_parse(maindir, class_path, tokenize=True)
            pbar.update(1)


def domain_data(src_lang, tgt_lang, tbx_path, class_path, cutoff, save_data, test_size=0.1):
    print("Building IATE term dataset by domains...")
    domain_dict = tbx2domain(tbx_path, src_lang, tgt_lang)
    domain_keys = [k for k, v in domain_dict.items() if len(v) > cutoff and len(v) < 75000]

    cmd = "/usr/bin/javac -cp " + class_path + " lib/*.java"
    os.system(cmd)

    with tqdm(total=len(domain_keys) - 1) as pbar:
        for dom in domain_keys:
            # Split into separate data set files
            maindir = os.path.join(save_data, str(dom))
            split_data(domain_dict[dom], maindir, test_size=test_size)

            # Dependency parsing
            dependency_parse(maindir, class_path, tokenize=True)
            pbar.update(1)


def main(opt):
    if opt.save_data:
        lang_list = eval(opt.lang_list)
        lang_data(opt.src_lang, lang_list, opt.tbx_path, opt.class_path, opt.save_data, opt.test_size, opt.pbar)
    if opt.save_domain:
        domain_data(opt.src_lang, opt.tgt_lang, opt.tbx_path, opt.class_path, opt.cutoff, opt.save_domain, opt.test_size)


if __name__ == '__main__':
    # try_debugger()

    parser = argparse.ArgumentParser(
        description='data_builder',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_data_opts(parser)
    opt = parser.parse_args()

    main(opt)
