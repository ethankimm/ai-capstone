## 2026-11-03 — TODO short title (coconut, run_id: 20261103-033009_coconut_v2-depth2-pilot_k8_coconut_c2m3)

**Goal:** TODO what question this run was trying to answer.
**Mechanism / model:** `coconut`, openai-community/gpt2 / /local-ssd/ktn48/lrm_work/coconut_runs/pilot_k8_coconut_c2m3/checkpoint_17, compute_steps=6.
**Data:** synthetic-depth2 test n=4000 seed=None; run seed=0.
**Hyperparams:** {'method': 'coconut', 'trainer': 'facebookresearch/coconut run.py @27273cb', 'config_path': '/local-ssd/ktn48/lrm_work/logs/configs/pilot_k8_coconut_c2m3.yaml', 'max_new_tokens': 16, 'batch_size': 64, 'test_json': '/local-ssd/ktn48/lrm_work/synthetic_data_v2/depth2_k8/test.json', 'decoding': 'greedy', 'train_config': {'project': 'coconut', 'save_path': '/local-ssd/ktn48/lrm_work/coconut_runs', 'name': 'pilot_k8_coconut_c2m3', 'only_eval': False, 'coconut': True, 'cot': False, 'no_thoughts': False, 'no_cot': False, 'c_thought': 2, 'epochs_per_stage': 3, 'max_latent_stage': 3, 'pad_latent_to_max': True, 'save_only_improve': False, 'uniform_prob': 0.0, 'model_id': 'openai-community/gpt2', 'load_model_path': '/local-ssd/ktn48/lrm_work/coconut_runs/gpt2_depth2_cot/checkpoint_3', 'seed': 0, 'resume': 3, 'bf16': False, 'train_path': '/local-ssd/ktn48/lrm_work/synthetic_data_v2/depth2_k8/train.json', 'val_path': '/local-ssd/ktn48/lrm_work/synthetic_data_v2/depth2_k8/valid_monitor.json', 'reset_optimizer': True, 'batch_size_training': 32, 'debug': False, 'gradient_accumulation_steps': 4, 'num_epochs': 17, 'lr': 0.0001, 'weight_decay': 0.01}}
**Command:** TODO
**Headline results:** `final_answer_accuracy=0.180`, `unparseable_rate=0.000`, `compute_steps=6`
**Interpretation:** TODO
**Gotchas hit:** TODO / none
**Caveats:** TODO
**Next:** TODO
