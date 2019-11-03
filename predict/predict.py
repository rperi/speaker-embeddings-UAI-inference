from unified_adversarial_invariance.model_configs import MODEL_CONFIGS_DICT
from unified_adversarial_invariance.datasets import DATASETS_DICT
from unified_adversarial_invariance.unifai import UnifAI_Config
from unified_adversarial_invariance.unifai import UnifAI
import kaldi_io

import argparse
import numpy
import os
import sys

def get_predictions_test(model_config, checkpoint_epoch,
                         remote_weights_path, data):
    # Create config
    config = UnifAI_Config()
    config.model_config = model_config
    config.remote_weights_path = remote_weights_path
   
    # Build model
    unifai = UnifAI(config, checkpoint_epoch)
    unifai.build_model_inference(checkpoint_epoch=checkpoint_epoch)
    unifai.model_inference.summary()
    # Set up data configuration

    # Get predictions and embeddings
    embeddings1, embeddings2 = unifai.get_embed(data)
    
    return embeddings1, embeddings2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'data_name', type=str,
        help='name of dataset'
    )
    parser.add_argument(
        'model_name', type=str,
        help='model name'
    )
    parser.add_argument(
        'feats_file', type=str,
        help='full path of the data(xvector array to be used as input for prediction) file'
    )
    parser.add_argument(
        'utts_file', type=str,
        help='full path of the utterances numpy array(corresponding to the feats_file above) file'
    )
    parser.add_argument(
        'weights_root', type=str,
        help='root dir of path containing model weights. Should contain "encoder-epoch.h5", For example encoder-00359.h5'
    )
    parser.add_argument(
        'checkpoint_epoch', type=int,
        help='checkpoint epoch'
    )
    parser.add_argument(
        'output_dir', type=str,
        help='output directory to save embeddings'
    )
    parser.add_argument(
        'save_kaldi_flag', type=int, default=1,
        help='whether to save the output embeddings as kaldi format scp files or not'
    )
    
    args = parser.parse_args()

    # Load data util and config for network-modules
    
    model_config = MODEL_CONFIGS_DICT[args.data_name].ModelConfig()
    
    if os.path.exists(args.feats_file):
        data = numpy.load(args.feats_file)
    else:
        sys.exit("input xvector feature numpy array doesn't exist. Please check feature numpy array path. Exiting!!!")

    embeddings1, embeddings2 = \
        get_predictions_test(
        model_config,  
        args.checkpoint_epoch, os.path.join(args.weights_root, args.model_name), data)

    # Save predictions and embeddings (for further use to visualize and compute accuracy)
    numpy.save(os.path.join(args.output_dir,"embed_1_test"), embeddings1)
    numpy.save(os.path.join(args.output_dir,"embed_2_test"), embeddings2)
    print('\nEmbeddings saved at %s \n' % args.output_dir)

    if args.save_kaldi_flag == 1:
        if not os.path.exists(args.utts_file):
            sys.exit("numpy array containing utterance information doesn't exist. Exiting!!!")

        # Save embeddings as kaldi ark files (for use in evaluations of eer)
        utts = numpy.load(args.utts_file)
        with open(os.path.join(args.output_dir, "e1_test.ark"), 'wb') as o:
        
            for idx, e in enumerate(embeddings1):
                utt = utts[idx]
                kaldi_io.write_vec_flt(o, e, key=utt)

        with open(os.path.join(args.output_dir, "e2_test.ark"), 'wb') as o:
        
            for idx, e in enumerate(embeddings2):
                utt = utts[idx]
                kaldi_io.write_vec_flt(o, e, key=utt)


if __name__ == '__main__':
    main()
