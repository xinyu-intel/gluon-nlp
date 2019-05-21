# Bert Quantization

## Bert Classification


```
# finetune
GLUE_DIR=glue_data python finetune_classifier.py --task_name MRPC --batch_size 32 --optimizer bertadam --epochs 3 --lr 2e-5

# export
python export/export.py --task classification --model_parameters ./output_dir/model_bert_MRPC_2.params --output_dir ./output_dir/ --seq_length 128

# fp32 inference
GLUE_DIR=./glue_data python symbolic_classifier.py --task_name MRPC --max_len=128 --model_parameters ./output_dir/classification --dev_batch_size 1 --pad 

# calibration
GLUE_DIR=./glue_data python symbolic_classifier.py --task_name MRPC --max_len=128 --model_parameters ./output_dir/classification --dev_batch_size 1 --pad --only-calibration 

# int8 inference
GLUE_DIR=./glue_data python symbolic_classifier.py --task_name MRPC --max_len=128 --model_parameters ./output_dir/classification-quantized-naive --dev_batch_size 1 --pad
```

## Bert QA

```
# finetune
python finetune_squad.py --optimizer adam --batch_size 12 --lr 3e-5 --epochs 2 --gpu

# export
python export/export.py --task question_answering --model_parameters ./output_dir/net.params --output_dir ./output_dir/ --seq_length 384

# fp32 inference
python symbolic_squad.py --test_batch_size 1 --max_seq_length=384 --model_parameters output_dir/question_answering

# calibration
python symbolic_squad.py --test_batch_size 32 --model_parameters output_dir/question_answering --only-calibration

# int8 inference
python symbolic_squad.py --test_batch_size 1 --max_seq_length=384 --model_parameters output_dir/question_answering-quantized-naive
```
