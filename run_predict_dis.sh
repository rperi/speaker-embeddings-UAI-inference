#!/bin/bash

source activate_env.sh
source path.sh

convert2numpy_flag=True
save_kaldi_flag=1

inp_dir=$PWD/example/input
out_dir_base=$PWD/example/output/
weights_dir=$PWD/example/model/

checkpoint_epoch=359

unique_id=xvec_ep${checkpoint_epoch}
out_dir=${out_dir_base}/${unique_id}
mkdir -p ${out_dir}

feats_scp=${inp_dir}/xvector.scp
numpy_feats_file=${inp_dir}/test_data.npy
numpy_utts_file=${inp_dir}/test_utt.npy

cd predict 
if [[ $convert2numpy_flag == True ]]; then
  cd ../pre_process
  python3 convert_scp2numpy.py ${feats_scp} ${inp_dir}
  cd ../predict/
  #exit

else
  echo "Script expects input xvectors as numpy array ${inp_dir}/test_data.npy"
fi

python3 predict.py xvector_uai xvector_uai_model \
$numpy_feats_file $numpy_utts_file $weights_dir $checkpoint_epoch \
${out_dir} ${save_kaldi_flag} || exit 1

echo "Succesfully extracted embeddings and saved to ${out_dir}"

######################################################################################
if [[ ${save_kaldi_flag} == 1 ]]; then
 copy-vector ark:${out_dir}/e1_test.ark ark,scp:${out_dir}/embed1_test.ark,${out_dir}/embed1_test.scp
echo "Final embedding e1 is in ${out_dir}/embed1_test.scp"

 copy-vector ark:${out_dir}/e2_test.ark ark,scp:${out_dir}/embed2_test.ark,${out_dir}/embed2_test.scp
echo "Final embedding e2 is in ${out_dir}/embed2_test.scp"

  if [ -f ${out_dir}/embed1_test.scp ]; then
    rm ${out_dir}/e1_test.ark
  fi

  if [ -f ${out_dir}/embed2_test.scp ]; then
    rm ${out_dir}/e2_test.ark
  fi
fi


