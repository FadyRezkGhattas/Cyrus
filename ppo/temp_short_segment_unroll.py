    def short_segment_unroll_(agent_state,
                            key,
                            inner_problem_length,
                            on_iteration,
                            truncation_length, # truncation_length is the length of steps to take in in the inner problem before doing meta-updates
                            start_time):
        def step(carry, step):
            agent_state, key, on_iteration, start_time = carry
            def reset_problem(k, agent_state):
                ppo_task = PPOTask(args)
                key1, key = jax.random.split(k)
                params, key = ppo_task.init(key)
                agent_state = VeloState.create(apply_fn=None, params=params, tx=agent_state.tx)
                on_iteration = 0
                return agent_state, on_iteration, key
            
            # If we have trained for longer than total inner problem length, reset the inner problem.
            jax.lax.cond(on_iteration >= inner_problem_length,
                        lambda k, agent_state: (reset_problem(k, agent_state)),
                        lambda k, agent_state: (agent_state, on_iteration, k),
                        key, agent_state)
            agent_state, key, step_losses = ppo_task.update(agent_state, key, start_time)

            # clip the loss to prevent diverging inner models
            step_losses = np.array(step_losses.flatten())
            cutoff = np.full_like(step_losses, 3.0, np.float64)
            step_losses = jnp.where(jnp.isnan(step_losses), cutoff, step_losses)
            losses = np.append(losses, step_losses)

            on_iteration += 1
        
            return (agent_state, key, on_iteration, start_time), losses
        
        (agent_state, key, on_iteration, start_time), losses = jax.lax.scan(step, (agent_state, key, on_iteration, start_time), (), length=truncation_length)
        return agent_state, key, on_iteration, start_time, losses