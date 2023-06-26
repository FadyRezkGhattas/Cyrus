import jax.numpy as jnp

evaluation_variables = {
    "Ackley": {"a": 20, "b": 0.2, "c": 2*jnp.pi, "dim" : 2, "eval_min": -32.768, "eval_max": 32.768},
    "Matyas": {"dim" : 2, "eval_min": -10, "eval_max": 10},
    "Booth": {"dim" : 2, "eval_min": -10, "eval_max": 10},
    "Rosenbrock": {"dim" : 2, "eval_min": -2.048, "eval_max": 2.048},
    "Michalewicz": {"dim" : 2, "eval_min": 0, "eval_max": jnp.pi},
    "Beale": {"eval_min": -4.5, "eval_max": 4.5},
    "Branin": {"a": 1, "b": 5.1/(4*jnp.pi**2), "c": 5/jnp.pi, "r": 6, "s": 10, "t": 1/(8*jnp.pi), "eval_min": -5, "eval_max": 15},
    "StyblinskiTang": {"dim" : 2, "eval_min": -5, "eval_max": 5},
    "Rastrigin": {"dim" : 2, "eval_min": -5.12, "eval_max": 5.12}
}
minimum_value = {
    "Ackley": 0,
    "Matyas": 0,
    "Booth": 0,
    "Rosenbrock": 0,
    "Michalewicz": -1.8013,
    "Beale": 0,
    "Branin": 0.397887,
    "StyblinskiTang": -39.16599*10,
    "Rastrigin": 0
}
minimum_coordinate = {
    "Ackley": [jnp.zeros(2)],
    "Matyas": [jnp.zeros(2)],
    "Booth": [jnp.array([1, 3])],
    "Rosenbrock": [jnp.ones(2)],
    "Michalewicz": [jnp.array([2.20, 1.57])],
    "Beale": [jnp.array([3, 0.5])],
    "Branin": [jnp.array([-3.14, 12.275]), jnp.array([3.14, 2.275]), jnp.array([9.42478, 2.475])],
    "StyblinskiTang": [jnp.array([-2.903534]*2)],
    "Rastrigin": [jnp.zeros(2)]
}