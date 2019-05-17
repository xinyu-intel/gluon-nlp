# Bert Classification


```
# finetune
GLUE_DIR=./glue_data python finetune_classifier.py --task_name MRPC --max_len=128 --epoch 1 --only_inference --model_parameters ./output_dir/model_bert_MRPC_2.params --dev_batch_size 1 --pad

# fp32 inference
GLUE_DIR=./glue_data python symbolic_classifier.py --task_name MRPC --max_len=128 --model_parameters ./output_dir/classification --dev_batch_size 1 --pad 

# calibration
GLUE_DIR=./glue_data python symbolic_classifier.py --task_name MRPC --max_len=128 --model_parameters ./output_dir/classification --dev_batch_size 1 --pad --only-calibration 

# int8 inference
GLUE_DIR=./glue_data python symbolic_classifier.py --task_name MRPC --max_len=128 --model_parameters ./output_dir/classification-quantized-naive --dev_batch_size 1 --pad
```

