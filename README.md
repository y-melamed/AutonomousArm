# AutonomousArm


Our fork from the original lerobot repo  - https://github.com/y-melamed/lerobot/tree/yanivTest

Our dataset of fifty put in the box tasks - https://huggingface.co/datasets/yanivmel1/new_dataset_cube

Our trained model - https://huggingface.co/yanivmel1/new_dataset_cube_080000



Build a robot, train a model for autonomus tasks and run a few Mechanistic interpretability experiments on vision-action Transformer

- we first have to build a robotic arm, we used the lerobot guide to order motors, and to three-d printing and assembling our robotic arm ans second arm for teleoperation. after assembling our setup looked like this <img>

- to obtain a dataset for training our model, we has to build a teleoperation system. here you can see a video of our teleoperation system where we can control the robot arm with an identical arm - https://drive.google.com/file/d/120DukbRjiXrmo3eBLwkWLlXfkgMk9tW1/view?usp=sharing


- next we collect dataset of fifty put in the box tasks, from five distinctive locations - our dataset available here.

- then we forked the lerobot repo <link> from huggingface and trained our model - the trained model is available here. 

- the resulting model succeded to learn grasping and placing objects in a box, and we can see that the model is able to generalize to unseen objects and locations.

<video - demos/autonomous_arm.mp4>


# mech interp experiments

- the model we trained is a transformer based model called ACT <path to paper https://arxiv.org/abs/2304.13705>

- we explore the model architecture, and hypothesi ....



- dataset for fine-tuning the model - https://huggingface.co/datasets/yanivmel1/fine_tune_1

- finetuned model - https://huggingface.co/yanivmel1/fine_tune_1_model




# AutonomousArm

Train a robotic arm to pick a cube and put in box autonomously and explore how the model makes decisions using mechanistic interpretability techniques.

---

## Project Overview

AutonomousArm is a research project focused on developing and analyzing a robotic arm controlled by Transformer based model. The goal is to understand not only how to achieve autonomous movement, but also how the underlying model represents and memorizes information.

---


## Model Architecture

- **Inputs:**  
  - Top and lateral camera images ($I_{\text{top}}, I_{\text{lat}}$)
  - Arm joint angles ($q$)
- **Feature Extraction:**  
  - Images processed by ResNet to produce feature vectors.
- **Tokenization:**  
  - 600 pixel tokens (300 per camera), arm state token, and condition token.
- **Transformer Encoding:**  
  - 3 encoder layers process all tokens for global information flow.
- **Action Vector Update:**  
  - Zero-initialized action vector updated via cross-attention and MLP to produce robot actions.

---

## 💡 Research Hypothesis

> The MLP layer in the decoder memorizes exact actions rather than generalizing, acting as a "lookup table" for robot movements.  
> Attention maps show only a subset of tokens receive high scores, and arm state information is absorbed by pixel tokens.

---

## 🧪 Validation Plan

1. **Probing Action Vector:**  
   Train linear classifiers to verify if the action vector encodes all relevant environmental details.
2. **Generalization Tests:**  
   Evaluate model performance on novel, unseen cube positions.
3. **Visualization:**  
   Use techniques like "Diffusion lens" to visualize hidden states and analyze stored information.

---

## 📚 Recommended Readings

- Interpreting visual features of the CLIP model (Goh et al., 2021)
- Logit Lens (nostalgebraist, 2024)
- Probing hidden states in Othello-GPT (Li et al., 2022)
- Where LLMs Store Information (Geva et al., 2023)
- Data Editing in LLMs (Meng et al., 2022)
- LLMs and Arithmetic (Nikankin et al., 2024)
- Physics of Language Models Series (Allen-Zhu, 2024)
- Sparse Autoencoders by Anthropic (2024)

---

## 📂 Repository Structure

- `demos/` — Videos and images of the robotic arm in action
- `experiments/` — Python scripts and notebooks for model training and analysis
- `Hypothesis/` — LaTeX source for research documentation
- `results/` — Output images and combined videos

---

## 📝 Citation

If you use this project, please cite the relevant papers listed above.

---

## 🖼️ Demo

See the `demos/` folder for example videos and images of the autonomous arm.

---

## 🔗 References

- [Project Paper (LaTeX)](Hypothesis/Hypothesis1.tex)
- [Mechanistic Interpretability Readings](mechanistic_interpretability_readings.pdf)

---
