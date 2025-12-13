### Setup

```bash
cd med-conformal

pip install -r requirements.txt

```

### Run

```bash
python main.py
```

## Configuration

Edit `configs/config.yaml` to customize:

```yaml
experiment:
  name: "your_experiment_name"
  seed: 42

data:
  size: 224       
  batch_size: 128     
  
model:
  architecture: "resnet18"
  pretrained: true
  dropout: 0.3

training:
  epochs: 50
  learning_rate: 0.001
  early_stopping:
    enabled: true
    patience: 10

conformal:
  alpha: 0.1    
  methods:      
    - naive
    - lac
    - raps_size
    - raps_temp
    - raps_adaptive

gradcam:
  n_samples_correct: 20
  n_samples_incorrect: 20
```