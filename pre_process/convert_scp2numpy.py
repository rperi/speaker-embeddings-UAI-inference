import kaldi_io
import numpy as np
import os
import sys

XVEC_DIM = 512

## This script converts a given xvector.scp file into numpy arrays test_data.npy
## It further stores the utterances as numpy arrays test_utt.npy

scp_file = sys.argv[1] # full path to the xvector scp file
out_dir = sys.argv[2] # output directory where the numpy arrays are saved

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

d = { key:mat for key,mat in kaldi_io.read_vec_flt_scp(scp_file) }

utts = []
xvec = np.empty((len(list(d.keys())), XVEC_DIM))
for idx, utt in enumerate(list(d.keys())):
    utts.append(utt)
    xvec[idx,:] = d[utt]

np.save(os.path.join(out_dir, 'test_data'), xvec)
np.save(os.path.join(out_dir, 'test_utt'), np.array(utts))
