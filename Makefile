# Isaac Lab 명령어 모음

train-anymal:
	python scripts/reinforcement_learning/rsl_rl/train.py \
	--task=Isaac-Velocity-Rough-Anymal-C-v0 \
	--num_envs 16

play-anymal:
	python scripts/reinforcement_learning/rsl_rl/play.py \
	--task=Isaac-Velocity-Rough-Anymal-C-Play-v0 \
	--num_envs 4 \
	--load_run $(RUN)

logs:
	ls ~/IsaacLab/logs/rsl_rl/anymal_c_rough/

collect-data:
	python scripts/data_collection/collect_anymal_data.py \
	--load_run $(RUN) \
	--num_steps 1000
