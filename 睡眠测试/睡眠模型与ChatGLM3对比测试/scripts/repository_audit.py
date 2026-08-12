#!/usr/bin/env python3
import json, re, hashlib
from pathlib import Path
root=Path('Reply-generate-sleep'); pkg=Path('repo_package'); nb=json.loads((root/'睡眠.ipynb').read_text())
outputs='\n'.join(''.join(o.get('text',[])) for c in nb['cells'] for o in c.get('outputs',[]))
weights=[]
for p in root.rglob('*'):
    if p.is_file() and p.suffix.lower() in {'.bin','.safetensors','.pt','.pth','.gguf','.ckpt'}: weights.append(str(p))
report={
 'repository_commit':'ed9aee2a626d825db5890a6016424105ac1a9f77','base_model':'zai-org/chatglm3-6b',
 'training_method':'PEFT LoRA, r=8, alpha=32, dropout=0.1; 3 epochs; lr=5e-5; input 128; output 256',
 'train_entry':'packages/chatglm3_colab_lora_package.zip::finetune_hf.py','inference_entry':'packages/chatglm3_colab_lora_package.zip::inference_hf.py',
 'domain_changes':['睡眠.ipynb','LoRA training package','500-example sleep dialogue dataset (440 train/60 dev)'],
 'weight_files_in_repository':weights,'adapter_available':bool(weights),
 'notebook_training_evidence':{'completed_165_steps':'100% 165/165' in outputs,'train_runtime_seconds':2850.1492 if '2850.1492' in outputs else None,'checkpoint_mentions':sorted(set(re.findall(r'checkpoint-\d+',outputs))),'eval_epoch3_rouge1':35.1372511627907 if '35.1372511627907' in outputs else None},
 'critical_finding':'Notebook records successful training and Drive checkpoints, but adapter files are absent from GitHub; the sleep model cannot be independently loaded from this repository alone.'}
Path('repository_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
