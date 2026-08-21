## Simplified customized version of ShinkaEvolve

Here group of summer intern students self-evolving reinforcement learning algorithm

### Setup

`pip install -r requirements.txt`, then install Atari ROMs (not covered by pip):
```
AutoROM --accept-license --install-dir "$(python3 -c "import multi_agent_ale_py, os; print(os.path.dirname(multi_agent_ale_py.__file__))")"
```