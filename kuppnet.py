import json

import click

from src import prepare_input, load_model, predict_and_dump, eval_and_dump

MODEL_PATHS = {"1": "models/ds1f1-.272",
               "2": "models/ds1f3-.278",
               "3": "models/ds4f1-.299"}
HPARAMS_PATHS = {"1": "models/hparams/model1.json",
                 "2": "models/hparams/model2.json",
                 "3": "models/hparams/model3.json"}
MODE_ACTION = {"eval": eval_and_dump,
               "predict": predict_and_dump,
               "train": ""}


@click.command()
@click.argument('mode', type=click.Choice(['eval', 'train', 'predict']))
@click.argument('input_seqs')
@click.option('-c', '--input_cls', type=click.File())
@click.option('-p', '--predictions', type=click.File('r'))
@click.option('-m', '--model', type=click.Choice(["1", "2", "3"]), default="3",
              help='number of model to be used')
@click.option('-d', '--device', help='GPU device number', default='0')
@click.option('-o', '--output_file', type=click.File('w'))
@click.option('-b', '--batch_size', type=int)
@click.option('-M', '--prediction_output_mode', default='tsv', type=click.Choice(['tsv', 'fasta']))
@click.option('-E', '--eval_output_mode', default='stats_only', type=click.Choice(['full', 'stats_only', 'tsv_only']))
@click.option('-w', '--window_size', type=int)
@click.option('-s', '--window_step', type=int)
@click.option('-t', '--threshold', type=float)
@click.option('-v', '--verbose', type=int, default=0)
@click.pass_context
def main(ctx, mode, input_seqs, input_cls, predictions, model,
         device, output_file, batch_size, prediction_output_mode,
         eval_output_mode, window_size, window_step, threshold, verbose):
    # TODO: update docs when train is finished
    # TODO: add models' descriptions
    """
    This is the main scrip to launch KUPPNet (Kinase Unspecific Phosphorylation Prediction Net).

    Three modes are available -- eval, predict and train (the last is unfinished).

    Tool uses TensorFlow library with Keras backend.

    Three pre-trained model are available, default is model №3.

    :param ctx: click context (for CLI arguments).
    Passed by default.
    :param mode:
    Script overall behaviour is completely dependent on which mode one chooses.
    One of three modes are available:
    1) predict
    Predicts phosphorylation sites using one of 3 models.
    Dumps output.
    2) eval
    Evaluates model's performance and dumps output.
    3) train
    TO BE COMPLETED
    :param input_seqs:
    Path to a file in fasta-like format with protein sequences.
    :param input_cls:
    Path to a file with id-true_positive_class_position pairs
    separated by space-like symbols.
    :param predictions:
    Path to a file with id-predicted_positive_class_position-score triples
    separated by space-like symbols.
    If provided in eval mode, no model will be compiled.
    :param model: number of model to be used.
    Three models are available:
    1)
    2)
    3)
    :param device: device to be used for computations.
    :param output_file: output will be dumped to a file
    specified by this param.
    If no file path is passed, output will be printed to stdout.
    :param batch_size: if not provided, keras default batch_size = 32
    will be used.
    :param prediction_output_mode: one tsv or fasta.
    If case of tsv id-predicted positive class pairs
    will be printed.
    In case of fasta predicted positive classes
    will be marked by lower-case letters
    :param eval_output_mode: one of
    1) tsv_only
    Tab-separated id, position, true value, predicted value
    will be printed for all potentially positive classes.
    1 means positive class, 0 means negative.
    2) stats_only
    Will calculate and print out accuracy, fnr, fpr, precision, recall, f1 and specificity.
    3) full
    Combination of 1 and 2.
    :param window_size: For rolling_window function.
    Since all models have their all requirements for input shape,
    leave default for predict and eval modes
    :param window_step: Unlike window_size can be safely overwritten.
    Determines window step to be used in rolling_window function
    :param threshold: value in (0, 1).
    :param verbose: 0 or 1. If 1 is provided all parameters will be printed
    to stdout.
    """
    ctx.params['input_seqs'] = input_seqs
    if verbose:
        print('\nFollowing parameters passed:',
              *sorted(ctx.params.items(), key=lambda x: x[0]), '\n', sep='\n')
    if ctx.params['mode'] == 'eval' and ctx.params['input_cls'] is None:
        raise ValueError('To use eval mode one must provide path to '
                         'a file with true classes using input_cls option')
    with open(HPARAMS_PATHS[model]) as f:
        hparams = json.load(f)
        if verbose:
            print("Loaded following default hparams:", '\n',
                  *sorted(hparams.items(), key=lambda x: x[0]),
                  'Working in {} mode'.format(ctx.params['mode']), sep='\n', )
    with open(input_seqs) as inp_seqs_handle:
        inp = prepare_input(inp_seqs_handle, hparams, ctx.params)
    compiled_model = load_model(MODEL_PATHS[model], hparams, device) if ctx.params['predictions'] is None else None
    if verbose and compiled_model is not None:
        print('\nModel has been compiled.\n', )
    action = MODE_ACTION[mode]
    action(model=compiled_model, inp=inp, cli_params=ctx.params, hparams=hparams)
    if verbose:
        print("\nJob is finished\n")


if __name__ == '__main__':
    main()
