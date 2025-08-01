# EAGLE3-MLLM

## File Structure
```bash
.
├── benchmarks
│   ├── __init__.py
│   └── run_SQA.py
├── cache
│   ├── dataset
│   ├── draft_model
│   └── target_model
├── demo.jpeg
├── demo.py
├── example
│   └── run_SQA.sh
├── README.md
├── requirements.txt
└── src
```

## 1. Clone the Repository

```bash
git clone https://github.com/Mimosa-Lin/EAGLE3-MLLM.git
cd EAGLE3-MLLM
```

## 2. Create and activate Conda environment

```bash
conda create -n eavlm python=3.10
conda activate eavlm
pip install -r requirements.txt
```

## 3. Download Model Checkpoints  
Download the following two model files into the corresponding directories:  

- Target model: [llava-hf/llava-1.5-7b-hf](https://huggingface.co/llava-hf/llava-1.5-7b-hf) → place under `cache/target_model` folder  
- Draft model: [LINMimosa/EAGLE_llava7B](https://huggingface.co/LINMimosa/EAGLE_llava7B) → place under `cache/draft_model` folder  

## 4. Run the demo.py
```bash
python -m demo
```

## Training
Training is performed using [SpecForge](https://github.com/sgl-project/SpecForge).

## References
- [SpecForge](https://github.com/sgl-project/SpecForge)
- [EAGLE](https://github.com/SafeAILab/EAGLE)
