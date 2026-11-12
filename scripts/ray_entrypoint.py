from __future__ import annotations

import json

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy


@ray.remote(num_cpus=1, num_gpus=1)
def identify() -> dict:
    return {
        "node": ray.get_runtime_context().get_node_id(),
        "gpu_ids": ray.get_gpu_ids(),
    }


ray.init(address="auto")
nodes = [node for node in ray.nodes() if node["Alive"]]
tasks = [
    identify.options(
        scheduling_strategy=NodeAffinitySchedulingStrategy(node["NodeID"], soft=False)
    ).remote()
    for node in nodes
]
print(json.dumps({"nodes": len(nodes), "tasks": ray.get(tasks)}, indent=2))


