"""
To use with (presumably trained) models which could be loaded through keras.models.load(filepath).
Loads data/hparams, prepares input, loads/compiles model, predicts/evaluates/trains, writes an output
Three working modes -- eval, predict, train
"""
import json

import click

from src import prepare_input, load_model, predict_and_dump

MODEL_PATHS = {"1": "models/ds1f1-.272",
               "2": "models/ds1f3-.278",
               "3": "models/ds4f1-.299"}
HPARAMS_PATHS = {"1": "models/hparams/model1.json",
                 "2": "models/hparams/model2.json",
                 "3": "models/hparams/model3.json"}
MODE_ACTION = {"eval": "",
               "predict": predict_and_dump,
               "train": ""}


@click.command()
@click.argument('mode', type=click.Choice(['eval', 'train', 'predict']))
@click.argument('input_seqs')
@click.option('-c', '--input_cls',
              help=('path to a file with true classes '
                    'with id(as in input_seqs)-position pairs separated by space(s) or tab(s)'))
@click.option('-m', '--model', type=click.Choice(["1", "2", "3"]), default="1",
              help='number of model to be used')
@click.option('-h', '--hparams', default='models/hparams/default_model_hparams.json')
@click.option('-d', '--device', help='GPU device number', default='0')
@click.option('-o', '--output_file', type=click.File('w'))
@click.option('-b', '--batch_size', type=int)
@click.option('-M', '--prediction_output_mode', default='tsv', type=click.Choice(['tsv', 'fasta']))
@click.option('-w', '--window_size', type=int)
@click.option('-s', '--window_step', type=int)
@click.option('-t', '--threshold', type=float)
@click.option('-v', '--verbose', type=int, default=0)
@click.pass_context
def main(ctx, mode, input_seqs, input_cls, model, hparams,
         device, output_file, batch_size, prediction_output_mode,
         window_size, window_step, threshold, verbose):
    if verbose:
        print('\nFollowing parameters passed:',
              *sorted(ctx.params.items(), key=lambda x: x[0]),
              '\nWorking in {} mode'.format(ctx.params['mode']), sep='\n')
    if ctx.params['mode'] == 'eval' and ctx.params['input_cls'] is None:
        raise ValueError('To use eval mode one must provide path to a file with true classes using input_cls option')
    with open(HPARAMS_PATHS[model]) as f:
        hparams = json.load(f)
        if verbose:
            print("\nLoaded following default hparams:",
                  *sorted(hparams.items(), key=lambda x: x[0]), sep='\n')
    with open(input_seqs) as inp_seqs_handle:
        inp = prepare_input(inp_seqs_handle, hparams, ctx.params)
    compiled_model = load_model(MODEL_PATHS[model], hparams, device)
    if verbose:
        print('\nModel has been compiled.\n', compiled_model.summary(), sep='\n')
    action = MODE_ACTION[mode]
    action(model=compiled_model, inp=inp, cli_params=ctx.params, hparams=hparams)
    if verbose:
        print("\nJob is finished\n")


if __name__ == '__main__':
    main()
