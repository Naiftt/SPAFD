# Suppressing Poisoning Attacks on Federated Learning for Medical Imaging
This repository includes the source code for MICCAI 2022 paper entitled: "Suppressing Poisoning Attacks on Federated Learning for Medical Imaging"


![diagram_paper](https://user-images.githubusercontent.com/50732592/175290166-63212932-10d3-4d8b-815a-0d35ea8c3078.png)
# Abstract 
Collaboration among multiple data-owning entities (e.g., hospitals) can accelerate the training process and yield better machine learning models due to the availability and diversity of data. However, privacy concerns make it challenging to exchange data while preserving confidentiality. Federated Learning (FL) is a promising solution that enables collaborative training through exchange of model parameters instead of raw data. However, most existing FL solutions work under the assumption that participating clients are honest and thus can fail against poisoning attacks from malicious parties, whose goal is to deteriorate the global model performance. In this work, we propose a robust aggregation rule called Distance-based Outlier Suppression (DOS) that is resilient to byzantine failures. The proposed method computes the distance between local parameter updates of different clients and obtains an outlier score for each client using Copula-based Outlier Detection (COPOD). The resulting outlier scores are converted into normalized weights using a softmax function, and a weighted average of the local parameters is used for updating the global model. DOS aggregation can effectively suppress parameter updates from malicious clients without the need for any hyperparameter selection, even when the data distributions are heterogeneous. Evaluation on two medical imaging datasets (CheXpert and HAM10000) demonstrates the higher robustness of DOS method against a variety of poisoning attacks in comparison to other state-of-the-art methods.
# Download Dataset (CheXpert and HAM10000)
[Chexpert](https://stanfordmlgroup.github.io/competitions/chexpert/) <br />
[HAM10000](https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000)

# Install dependencies
function test() {
  console.log("pip install -r requirements.txt");
}
```

