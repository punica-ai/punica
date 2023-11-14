# Example: Finetune & Convert weight to Punica format

In this example, we will first use [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) to finetune [Llama2-7B](https://huggingface.co/meta-llama/Llama-2-7b-hf) on three datasets: [gsm8k](https://huggingface.co/datasets/gsm8k), [sqlctx](https://huggingface.co/datasets/b-mc2/sql-create-context), [viggo](https://huggingface.co/datasets/GEM/viggo). Then, convert the PEFT weight to Punica format. Finally, we run each model using Punia.

## Download finetuned weight

If you want to skip the finetuning process, you can download the finetuned weights:

```bash
# CWD: project root
mkdir -p model
git lfs install
git clone https://huggingface.co/abcdabcd987/gsm8k-llama2-7b-lora-16 model/gsm8k-r16
git clone https://huggingface.co/abcdabcd987/sqlctx-llama2-7b-lora-16 model/sqlctx-r16
git clone https://huggingface.co/abcdabcd987/viggo-llama2-7b-lora-16 model/viggo-r16
```

## Finetune on local GPU

If you prefer to finetune by yourself:

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git --branch v0.2.2 examples/finetune/LLaMA-Factory

python examples/finetune/create-finetune-data.py --preset gsm8k
python examples/finetune/create-finetune-data.py --preset sqlctx
python examples/finetune/create-finetune-data.py --preset viggo

bash examples/finetune/finetune.sh gsm8k
bash examples/finetune/finetune.sh sqlctx
bash examples/finetune/finetune.sh viggo
```

## Convert weight to Punica format

```bash
python -m punica.utils.convert_lora_weight model/gsm8k-r16/adapter_model.bin model/gsm8k-r16.punica.pt
python -m punica.utils.convert_lora_weight model/sqlctx-r16/adapter_model.bin model/sqlctx-r16.punica.pt
python -m punica.utils.convert_lora_weight model/viggo-r16/adapter_model.bin model/viggo-r16.punica.pt
```

## Test run

```bash
gsm8k_prompt=$'<<SYS>>\nAnswer the following Grade School Math problem.\n<</SYS>>\n[INST] A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take? [/INST]\n'
sqlctx_prompt=$'<<SYS>>\nGenerate a correct SQL query from the following database schema.\nCREATE TABLE student_course_registrations (student_id VARCHAR, registration_date VARCHAR); CREATE TABLE students (student_details VARCHAR, student_id VARCHAR)\n<</SYS>>\n[INST] What is detail of the student who most recently registered course? [/INST]\n'
viggo_prompt=$'<<SYS>>\nGenerate a description based on the following representation.\n<</SYS>>\n[INST] verify_attribute(name[Metal Gear Solid 3: Snake Eater], release_year[2004], rating[excellent], genres[action-adventure, shooter, tactical]) [/INST]\n'

python examples/textgen_lora.py --lora-weight model/gsm8k-r16.punica.pt --prompt "$gsm8k_prompt"
python examples/textgen_lora.py --lora-weight model/sqlctx-r16.punica.pt --prompt "$sqlctx_prompt"
python examples/textgen_lora.py --lora-weight model/viggo-r16.punica.pt --prompt "$viggo_prompt"
```

Reference outputs:

```
It takes 2/2=<<2/2=1>>1 bolt of white fiber
So the total amount of fabric is 2+1=<<2+1=3>>3 bolts of fabric
#### 3

SELECT T2.student_details FROM student_course_registrations AS T1 JOIN students AS T2 ON T1.student_id = T2.student_id ORDER BY T1.registration_date DESC LIMIT 1

You mentioned that you greatly enjoyed Metal Gear Solid 3: Snake Eater. Would you say you're a big fan of action-adventure games from 2004 involving shooting and tactical gameplay?
```
