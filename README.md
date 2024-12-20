# Facial Nerve Paralysis Assessment and Analysis

This repository represents the culmination of our summer research project, where I worked on using computer vision and machine learning techniques to diagnosing and determining the severity of Bell's Palsy, as well as analyzing our methods effectiveness.

The follow-up project repo can be found [here](https://github.com/i-am-nolan25/Bells-Palsy-ML-Research).

To recreate the environment I used for this project, get [bp.yml](./conda_environments/bp.yml) in the **conda_environments** folder and run the following command:

    conda env create -f bp.yml


## Researcher

- Bingnan Huo

## Mentors

- Dr. Joshua Stough
- Dr. Keith Buffinton
- Dr. Arun Gadre

## Sponsored by

- [Bucknell University](https://www.bucknell.edu)
- [Geisinger](https://www.geisinger.org/)

## Research Presentation

This research was presented at the 11th [Susquehanna Valley Undergraduate Research Symposium (SVURS 2022)](https://www.geisinger.org/susquehanna-valley-undergraduate-research-symposium). 

- [Poster](https://eg.bucknell.edu/~jvs008/research/palsy/2022_Huo_Facial_Palsy.pdf)
- [Report](https://eg.bucknell.edu/~jvs008/research/palsy/2022_Huo_Report.pdf)
- [Presentation](https://www.youtube.com/watch?v=XmrpCGHMlbU)


## Background

Facial nerve paralysis (FNP) is a medical condition that affects the ability of individuals to control certain facial muscles. One of the most common forms of FNP is Bell's Palsy, which affects approximately 4 out of 10,000 people in the United States each year. People with Bell's Palsy experience one or, in rare cases, both sides of their face becoming unresponsive. The exact cause of Bell's Palsy is unknown, and it typically takes up to six months or longer for a patient to recover after being affected.

In the past, all diagnoses of FNP were made by doctors based on their subjective judgement and professional experience. However, previous research [[1]](#References) has shown that observer bias is common when FNP patients are seen and diagnosed by clinicians, and that a machine learning (ML) based approach found less facial asymmetry in severe FNP patients and more asymmetry in healthy faces than clinicians.

The goal of the research project discussed in this report is to develop software that uses computer vision and machine learning techniques to grade the severity of FNP and assist in the diagnosis and recovery tracking of patients with FNP using the House-Brackmann scale. The aim is to provide doctors and patients with another tool that can be used for reference and potentially reduce bias in diagnosis.

To achieve this goal, the research project will follow a framework similar to that used in other studies with similar goals. This includes:

- Using an ML-based landmark localizer to determine the locations of key facial anatomical points (landmarks) on an image of the face.
- Correcting for head tilt or perspective distortion using geometric or ML algorithms.
- Calculating asymmetry in the face using some algorithmic method.
- Using an algorithmic or machine learning-based approach to translate the asymmetry measurements into classifications.

## Literature Review

These are the most important studies we used in our work:

- Guarin [[2]](#References) published Emotrics, the landmark predictor various studies have used. We are also using Emotrics. Emotrics uses dlib to predict for 68 key facial landmark points. It included several trained models, including one trained using the iBUG 300-W dataset and the MEEI facial palsy dataset [[3]](#References). We will be using the MEEI model as it is trained on the MEEI dataset [[3]](#References) that consists of 60 patients with a spectrum of types and severities of FNP. Guarin wrote Emotrics with the intention of creating an app that would help doctors measure facial asymmetries. Thus, Emotrics has a GUI, built with PyQt5. However, we are only interested in using the part of code that would extract the facial landmarks from input images.

- Bandini [[4]](#References) explained that having specific patient-population-trained models or fine-tuning pretrained models (using the general population) will lead to lower prediction errors measured by nRMSE%.

- Gemma Parra-Dominguez’s study [[5]](#References) is fundamental to our work. We are mainly incorporating his proposed framework and adding the severity grading functionality in the final step. Gemma proposes to use 51 landmark points instead of 68 as the jaw line is not the main concern for evaluating illness or severity grading of facial palsy. Gemma also defined a set of distances and features, found through an ML program. Gemma’s work used a Multilayer Perceptron Network to differentiate between healthy & ill images.

## Packages Used

We utilize a variety of technologies in our research project, including, but not limited to, the following:

- **Image Processing**: OpenCV, scikit-image, Dlib
- **Data Processing**: pandas
- **Machine Learning**: scikit-learn
- **Visualizations**: Matplotlib, Jupyter Notebook
- **Miscs**: NumPy, SciPy


## References

- [1] Miller, Matthew Q., et al. “The Auto-eFACE: Machine Learning-Enhanced Program Yields Automated Facial Palsy Assessment Tool.” Plastic and reconstructive surgery vol. 147, 2 (2021): 467-474. doi:10.1097/PRS.0000000000007572

- [2] Guarin, Diego L., et al. “Toward an Automatic System for Computer-Aided Assessment in Facial Palsy.” Facial plastic surgery & aesthetic medicine vol. 22, 1 (2020): 42-49. doi:10.1089/fpsam.2019.29000.gua

- [3] Greene, Jacqueline J., et al. “The spectrum of facial palsy: The MEEI facial palsy photo and video standard set.” The Laryngoscope vol. 130, 1 (2020): 32-37. doi:10.1002/lary.27986

- [4] Bandini, Andrea, et al. “A New Dataset for Facial Motion Analysis in Individuals With Neurological Disorders.” IEEE journal of biomedical and health informatics vol. 25, 4 (2021): 1111-1119. doi:10.1109/JBHI.2020.3019242

- [5] Parra-Dominguez, et al. “Facial Paralysis Detection on Images Using Key Point Analysis.” Appl. Sci. 2021, 11, 2435. https://doi.org/10.3390/app11052435

## License 

The project is available as open source under the terms of the [MIT License](LICENSE).

