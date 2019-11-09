# speaker_embeddings_UAI_inference
    Code for extracting speaker embeddings using x-vector as input (comes with pre-trained model)
    Based on unsupervised adversarial invariance technique
    Submitted to IEEE International Conference on Acoustics, Speech and Signal Processing 2020
    Extracts 2 disentangled representations from the x-vectors
        embed1 containing speaker information
        embed2 containing all other information
        
[arxiv submission](https://arxiv.org/pdf/1911.00940.pdf)

## Dependencies
    1) python (verified with v3.6)
    2) Tensorflow-gpu (verified with v1.8.0)
    3) keras (verified with v2.1.2)
    4) kaldi (verified with v5.5)

## Example usage

    STEP 1. 
        IF python environment exists, then activate your own envirnoment before proceeding to STEP 2
        To create a new conda environment using provided environment.yml, run "source activate_env.sh" file (requires anaconda v3)

    STEP 2. Modify KALDI_ROOT in path.sh to the kaldi root directory in your local machine

    STEP 3. After running STEP 1 and STEP 2 above, run the following "example" command
    bash run_predict_dis.sh $PWD/example/input $PWD/example/output/ $PWD/example/model/xvector_uai_model/encoder-00359.h5 True 1

## Inputs
    Takes 5 arguments
        inp_dir: Directory containing xvectors either as "xvector.scp" or "test_data.npy" and "test_utt.npy"
        
        out_dir: Directory where the output embeddings are saved
        
        weights_path: Path to encoder model
        
        convert2numpy_flag ("True" or "False"): Flag to convert the xvectors from kaldi format to python numpy array or not
            HAS to be set to True if "test_data.npy" and "test_utt.npy" donot exist in inp_dir
        
        SAVE_AS_KALDI ("1" or "0"): Flag to save the output embeddings in kaldi format
            If set to 1, embeddings are saved as embed1_test.scp and embed2_test.scp as well as numpy arrays
            If set to 0, embeddings are saved only as numpy arrays embed1_test.npy and embed2_test.npy

## Outputs
    If SAVE_AS_KALDI is 0: Saves embeddings in out_dir as embed1_test.npy and embed2_test.npy
    If SAVE_AS_KALDI is 1: Saves embeddings in out_dir as embed1_test.npy, embed1_test.scp and embed2_test.npy,embed2_test.scp
