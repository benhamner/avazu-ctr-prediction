PROJECT_DIR = $(DataPath)/Kaggle/avazu-ctr-prediction
RAW_DIR     = $(PROJECT_DIR)/Raw
WORKING_DIR = $(PROJECT_DIR)/Working

head-train:
	julia src/head_train.jl $(RAW_DIR)
