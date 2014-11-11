PROJECT_DIR = $(DataPath)/Kaggle/avazu-ctr-prediction
RAW_DIR     = $(PROJECT_DIR)/Raw
WORKING_DIR = $(PROJECT_DIR)/Working

head-train:
	julia src/head.jl $(RAW_DIR)/train_rev2.gz

head-test:
	julia src/head.jl $(RAW_DIR)/test_rev2.gz

lr:
	julia src/lr.jl $(RAW_DIR)
