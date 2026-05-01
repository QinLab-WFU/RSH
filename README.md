# RSH
Source code for ICML 2026 paper “Robust Self-reflective Hashing for Cross-modal Retrieval with Noisy Label”

# Training
### **Processing dataset**
Refer to [DSPH](https://github.com/QinLab-WFU/DSPH).

### **Download the CLIP pretrained model**
The pretrained model will be found in the 30 lines of [CLIP/clip/clip.py](https://github.com/openai/CLIP/blob/main/clip/clip.py). This code is based on the "ViT-B/32".
You should copy ViT-B-32.pt to the directory of IDGH.

### **Start**
```bash
python main.py
