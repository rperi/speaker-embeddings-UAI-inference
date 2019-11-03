# speaker_embeddings_UAI_predict
Code for extracting speaker embeddings using x-vector as input (comes with pre-trained model)

## Dependencies

    python (verified with v3.6)
    Tensorflow-gpu (verified with v1.8.0)
    keras (verified with v2.1.2)
    kaldi

## Usage

    STEP 1. 
        IF python environment exists, then activate your own envirnoment before proceeding to STEP 2
        To create a new conda environment using provided environment.yml, run "source activate_env.sh" file (requires anaconda v3)

    STEP 2. Modify KALDI_ROOT in path.sh to path in your local machine

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
