Please run 
```
python eval/eval_se/run_evaluation.py --ensemble  --task-dir benchmark/tasks_v4 --workers 20
```

Workers 20 usually works well for me, please make sure to specify --ensemble

It will save results to `eval/eval_se/results` in a json format.