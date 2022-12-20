# Codebase for NDR's experiments

We leveraged the codebase used in the paper "The Neural Data Router: Adaptive Control Flow in Transformers Improves Systematic Generalization". 

Paper: https://arxiv.org/abs/2110.07732


## Setup

This project requires Python 3 (tested with Python 3.8 and 3.9) and PyTorch 1.8.

```bash
pip3 install -r requirements.txt
```

### Running experiments locally

It is possible to run single experiments with Tensorboard without using Weights and Biases. This is intended to be used for debugging the code locally.
  
If you want to run experiments locally, you can use ```run.py```:

```bash
./run.py sweeps/ctl_ndr.yaml
```
To run fourier+NDR on simple arithmetic, run ```python3 run.py sweeps/transformer_control_flow/simple_arithmetics_fourier.yaml```
To run fourier+NDR on listops, run ```python3 run.py sweeps/transformer_control_flow/listops_big_lowlr_fourier_ndr.yaml```

If the sweep in question has multiple parameter choices, ```run.py``` will interactively prompt choices of each of them.

The experiment also starts a Tensorboard instance automatically on port 7000. If the port is already occupied, it will incrementally search for the next free port.

Note that the plotting scripts work only with Weights and Biases.


# BibText
```
@article{csordas2021neural,
      title={The Neural Data Router: Adaptive Control Flow in Transformers Improves Systematic Generalization}, 
      author={R\'obert Csord\'as and Kazuki Irie and J\"urgen Schmidhuber},
      journal={Preprint arXiv:2110.07732},
      year={2021},
      month={October}
}

@inproceedings{lee-thorp-etal-2022-fnet,
    title = "{FN}et: Mixing Tokens with {F}ourier Transforms",
    author = "Lee-Thorp, James  and
      Ainslie, Joshua  and
      Eckstein, Ilya  and
      Ontanon, Santiago",
    booktitle = "Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jul,
    year = "2022",
    address = "Seattle, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.naacl-main.319",
    doi = "10.18653/v1/2022.naacl-main.319",
    pages = "4296--4313",
    abstract = "We show that Transformer encoder architectures can be sped up, with limited accuracy costs, by replacing the self-attention sublayers with simple linear transformations that {``}mix{''} input tokens. Most surprisingly, we find that replacing the self-attention sublayer in a Transformer encoder with a standard, unparameterized Fourier Transform achieves 92-97{\%} of the accuracy of BERT counterparts on the GLUE benchmark, but trains 80{\%} faster on GPUs and 70{\%} faster on TPUs at standard 512 input lengths. At longer input lengths, our FNet model is significantly faster: when compared to the {``}efficient Transformers{''} on the Long Range Arena benchmark, FNet matches the accuracy of the most accurate models, while outpacing the fastest models across all sequence lengths on GPUs (and across relatively shorter lengths on TPUs). Finally, FNet has a light memory footprint and is particularly efficient at smaller model sizes; for a fixed speed and accuracy budget, small FNet models outperform Transformer counterparts.",
}
```
