Augmenting Cross-lingual Terminologies with Tree-to-Sequence Neural Machine Translation
======

*Companion source code*

## Requirements

All dependencies can be installed via:

```bash
pip install pipenv
pipenv install
```

## Quickstart

### 1 Download IATE data archive

We start by downloading the IATE term bank archive, along with the necessary CoreNLP libraries.

```bash
bash tools/download.sh
```

### 2 Read and build raw data files

We write raw text source and target term files from the IATE archive, which is saved in the TermBase eXchange (TBX) format.

```bash
python tools/build_data.py --tbx-path data/IATE_export.tbx --save-data data
```

### 3 Preprocess the data

```bash
lg=fr  # define target language

python preprocess.py --train-src data/en_"$lg"/train/source.tok --train-tgt data/en_"$lg"/train/target.txt --train-src-parse data/en_"$lg"/train/source.parent --valid-src data/en_"$lg"/valid/source.tok --valid-tgt data/en_"$lg"/valid/target.txt --valid-src-parse data/en_"$lg"/valid/source.parent --pre-word-vecs-src fasttext/crawl-300d-2M.vec --pre-word-vecs-tgt fasttext/cc."$lg".300.vec --save-data data/en_"$lg"
```

We create separate archives for train and validation sets of the source and target terms data, along with their dependency parse files. A vocab object (with vocabulary built from the term sets) is also created with the inclusion of pretrained word embeddings specified.

So we generate three files:

  - `data/en_$lg.train.py`: serialized PyTorch file containing training data
  - `data/en_$lg.valid.py`: serialized PyTorch file containing validation data
  - `data/en_$lg.vocab.py`: serialized PyTorch file containing vocabulary data

### 4 Training

We train the model, keeping in general default parameters.

```bash
lg=fr  # define target language

python train.py --data data/en_"$lg"_tree --gpu-ranks 0 --encoder-type tree --decoder-type tree --tree-combine --checkpoint-dir checkpoint --tensorboard-dir tensorboard
```

### 5 Translate

Finally we do beam search translation decoding with the trained model checkpoints.

```bash
lg=fr  # define target language

python translate.py --model checkpoint_020000.pt --src data/en_"${LG[$i]}"/test/source.tok --src-parse data/en_"${LG[$i]}"/test/source.parent --tgt data/en_"${LG[$i]}"/test/target.txt --output pred.txt --gpu 0 --replace-unk
```

This will output predictions into `pred.txt`.

## License

The codebase are rewritten from work licensed under the [MIT license](https://github.com/OpenNMT/OpenNMT-py/blob/master/LICENSE.md), copyright (c) 2017-Present OpenNMT (https://github.com/OpenNMT/OpenNMT-py).

Parts of the codebase are rewritten bgom work licensed under the [MIT license](https://github.com/dasguptar/treelstm.pytorch/blob/master/LICENSE), copyright (c) 2017 Riddhiman Dasgupta (https://github.com/dasguptar/treelstm.pytorch).

To the extent possible under law, the author(s) have dedicated all copyright and related and neighboring rights to this software to the public domain worldwide. This software is distributed without any warranty. You should have received a copy of the [CC0 Public Domain Dedication](LICENSE) along with this software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>
