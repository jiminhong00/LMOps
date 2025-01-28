#need to install following
# conda install jsonlines
# python -m spacy download en_core_web_sm
conda install datasets
pip install faiss-gpu-cu12 ///
conda install scikit-learn

#uprise/src/metric -> from datasets import load_metric // has been deprecated
#use import evaluate // def compute_bleu: BLEU = evaluate.load("bleu")
conda install evaluate 

