KUPPNet
==============================

KUPPNet (Kinase Unspecific Phosphosites' Prediction Net) is CNN-RNN
based phosphorylation site prediction tool.

For now, it's available in two modes:

1) *predict*.
basic usage example
```python kuppnet.py predict path/to/fasta/file```
2) *eval*.
basic usage example
```python kuppnet.py eval path/to/fasta/file --input_cls path/to/true/classes```
where the `path/to/true/classes` is a path to tsv-like file with
id--true_positive_class_position pairs.

For prediction one of 3 models are used, trained on separate data sets.
Default model is model №3.

Article with description of models' architectures and data sets is
submitted.

More information about main script usage can be obtained using
`python3 kuppnet.py --help` command.

If you have any questions, feel free to ask (edikedikedikedik@gmail.com).
