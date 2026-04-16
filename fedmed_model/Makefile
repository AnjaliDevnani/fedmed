# Makefile — run everything with simple commands

.PHONY: install data train-central fl-server fl-simulate evaluate audit predict tensorboard clean

install:
	pip install -r requirements.txt

# Step 1: Download and prepare data
data:
	python preprocess.py
	python data_split.py

# Step 2: Train centrally (baseline verification)
train-central:
	python train_central.py

# Step 3: Run FL simulation (3 terminals needed)
fl-server:
	python fl_server.py

fl-hospital-A:
	python fl_client.py --hospital A

fl-hospital-B:
	python fl_client.py --hospital B

fl-hospital-C:
	python fl_client.py --hospital C

# Alternative Step 3: Run all FL simulation in one terminal
fl-simulate:
	python train_federated.py

# Step 4: Evaluate global model
evaluate:
	python evaluate.py

# Step 5: Check privacy guarantees
audit:
	python privacy_audit.py

# Run single prediction
predict:
	python predict.py --image ./sample/chest_xray.jpg

# TensorBoard
tensorboard:
	tensorboard --logdir ./logs --port 6006

# Clean
clean:
	rm -rf checkpoints/ logs/ results/
	find . -name "__pycache__" -type d -exec rm -rf {} +
