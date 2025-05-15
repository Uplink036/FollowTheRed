move: ## Run the swarm command
	cd swarm & ./move.sh

simulation_run: ## Run the python program
	python3 ./src/Jetbot_objects_move.py

issac:
	python -m venv env_isaacsim
	env_isaacsim\Scripts\activate
	pip install isaacsim[all]==4.5.0 --extra-index-url https://pypi.nvidia.com

help: ## Show this help
	@grep -E '^[.a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
