# Explainable AI For Diabetic Retinopathy And Screening In Rural India

An experimental deep learning project that grades diabetic retinopathy (DR) from retinal fundus photographs and shows **which parts of the image influenced the prediction** (Grad-CAM heatmaps), so that a doctor can cross-check the result.

Built for **Smart India Hackathon 2026** (problem statement SIH26038) by a team of five. This repository contains the **model code**: data preparation, training, evaluation, Grad-CAM and a command-line prediction script.

> **Important:** this is a student research prototype. It is **not a medical device**, it has **not been clinically validated**, and it must not be used to diagnose or treat anyone. It is meant to support, never replace, an ophthalmologist.

---

## Why this project

Diabetic retinopathy can cause blindness if it is found late, and many people in rural India do not have easy access to an eye specialist. A screening tool that flags likely cases and explains its reasoning could help doctors prioritise patients. This project is a first step toward that idea, and it documents honestly where it still falls short.

## What the system does

1. **Image quality check:** blurry or badly exposed images are rejected, because a poor photo should not receive a grade.
2. **Preprocessing:** crop black borders, resize to 300 x 300, and apply Ben Graham preprocessing (subtract a heavily blurred copy of the image to boost local detail such as small lesions).
3. **Two-stage cascade**
   - **Stage 1:** No DR vs DR.
   - **Stage 2:** severity grading of DR images into Mild, Moderate, Severe and Proliferative DR.
4. **Explainability:** Grad-CAM heatmaps highlight the retinal regions that influenced the prediction.

The five classes are: `No_DR`, `Mild`, `Moderate`, `Severe`, `Proliferative_DR`.

## Model details

- Backbone: **DenseNet121** (ImageNet weights) in TensorFlow / Keras, with global average pooling, batch normalisation, dropout and a 256-unit dense layer.
- Training: head training with a frozen backbone, then fine-tuning of the top layers at a low learning rate.
- **Ordinal loss**, so that mistakes between distant grades (for example Moderate predicted as Proliferative) cost more than mistakes between neighbouring grades.
- Class imbalance handled with oversampling of rare classes and light class weights.
- Optional test-time augmentation (flips and rotation) when evaluating.

## Datasets

The model was trained on a merged dataset made from:

- **DDR** dataset
- **IDRiD** dataset
- **APTOS 2019 dataset 

An additional dataset (Messidor-2) was tried during experiments, but it lowered validation accuracy, so it is **not** part of the final training data.

**The datasets are not included in this repository.** They are large and have their own licences. Please download them from their official sources.

## Trained models

The stage 1 and stage 2 models used for the v1 results below are in `saved_models/`:

- `cascade_stage1_final_final_A_fixed.keras`
- `cascade_stage2_final_final_A_fixed.keras`

These were trained on public datasets, so check each dataset's licence before reusing the models.

## Results

Evaluated on a held-out **validation split (998 images, a random 20% of the merged data)**, using the two-stage cascade:

| Version | Accuracy | Macro F1 | No_DR F1 | Mild F1 | Moderate F1 | Severe F1 | Proliferative F1 |
|---|---|---|---|---|---|---|---|
| v1 (first cascade) | 81.8% | 0.727 | 0.961 | 0.611 | 0.719 | 0.470 | 0.873 |
| v2 (image-quality augmentation + learning from mistakes) | 83.7% | 0.708 | 0.965 | 0.568 | 0.779 | 0.366 | 0.862 |

How to read this:

- Overall accuracy rose a little in v2, but macro F1 fell, because **Severe** images were more often predicted as **Moderate** (v1 caught 66% of Severe cases, v2 only 28%).
- For a screening tool, **under-grading severe disease is the most worrying error**, so v2 is not simply "better". The v1 models are the ones provided here.
- Most mistakes are between **neighbouring grades**, which is typical for DR grading.
- **These numbers come from a validation split drawn from the same sources as the training data.** They do not show how the model performs on new cameras, clinics or patients.

## Known limitations

- **No external validation.** On retinal images from outside the training sources (for example images found on the web), predictions were unreliable and often biased toward Proliferative DR. This is *domain shift*: different cameras, image sizes, colours and compression.
- **Class imbalance.** There are far more No DR and Proliferative images than Mild and Severe ones, so the rare classes are harder.
- **Severe DR is the weakest class**, and it is the class where a missed case matters most.
- **Labels were not re-checked by a doctor.** The public datasets come with their own labels, but being on a public platform does not make a dataset clinically verified. Grading is also subjective, and different graders can disagree.
- **The model always gives an answer.** It has no "I am not sure" option for unfamiliar images.
- **No patient information** (age, history, glasses, refractive error) is used.
- Trained on CPU only, so only a few training configurations could be tried.

## Future work

**1. Better and more diverse data**
- Partner with eye clinics to collect fundus images **with patient consent and ethics approval**, covering different cameras, ages and regions.
- Have **ophthalmologists verify every image and label** instead of trusting a public dataset's labels, and record disagreements between graders. A doctor-verified dataset could also support a proper **research paper**.
- Collect more Mild and Severe examples to reduce imbalance.

**2. Patient factors**
- **Age:** the retina and the risk of disease change with age, so group images by age band and test whether the model behaves differently across groups.
- **Glasses and refractive error (myopia / high power):** myopic eyes can look different in fundus photos. Record this information and check whether accuracy changes for these patients.
- This needs enough data for every group, which is why it depends on the clinic data collection above.

**3. Bias and fairness**
- Report performance separately for each camera type, age group, sex and region, not only overall.
- Check whether the model is biased toward the majority classes, and fix it with better sampling, loss functions or more data.

**4. Modelling**
- Add an **uncertainty option** ("not confident, please review") so unfamiliar or low-quality images are not graded.
- Try newer backbones such as EfficientNetV2, and calibrate the confidence scores.
- Train on a GPU to allow longer training and a proper hyper-parameter search.
- Combine the image with patient factors (age, refractive error) once the data exists.

**5. Evaluation and safety**
- Test on **independent external datasets** and on prospectively collected clinic images.
- Choose sensitivity and specificity targets together with clinicians, and give extra weight to not missing Severe and Proliferative cases.
- Follow the regulatory and ethical requirements for medical software before any real-world use.

## Project structure

```
src/
  config.py          paths and hyper-parameters
  data_loader.py     image loading, preprocessing, oversampling, augmentation
  model.py           DenseNet121 classifier
  ordinal_loss.py    ordinal loss function
  train.py           single-stage training
  cascade_train.py   two-stage cascade training and evaluation
  gradcam.py         Grad-CAM heatmaps
  utils.py
predict.py           predict one image with the trained cascade
eval_old.py          evaluate a saved cascade on the validation split
merge_ddr.py, merge_idrid.py, organize_data.py, verify_dataset.py   dataset preparation
saved_models/        the two trained cascade models (see "Trained models")
```

## How to run

1. Install Python 3.11 and the packages in `requirements.txt` (TensorFlow 2.15).
2. Download the datasets and arrange them as:
   ```
   data/raw/train/<image_id>.png
   data/train_labels_combined.csv      (tab-separated, columns: id_code, diagnosis)
   ```
   Diagnosis values: 0 = No DR, 1 = Mild, 2 = Moderate, 3 = Severe, 4 = Proliferative DR.
3. Train the cascade: `python -m src.cascade_train`
4. Predict one image: `python predict.py path/to/image.png`

## Team and acknowledgements

Smart India Hackathon 2026 team project. The model, training, evaluation and Grad-CAM code in this repository were written by **Ankita Rath**. The front end and back-end integration were built by teammates and are not included here.

Datasets: DDR, IDRiD and the dataset provided by Aravind Eye Hospital. Method references: Ben Graham's preprocessing from the Kaggle Diabetic Retinopathy competition, and Grad-CAM (Selvaraju et al.).

## Disclaimer

For research and education only. Not for clinical use.