import re

with open("06_dreamer_v3.py", "r") as f:
    content = f.read()

prefix, loop_part = content.split("print(\"Starting Training Loop...\")")

new_loop = """print("Starting Training Loop...")
    global_step = 1000
    ep_return = 0
    ep_len = 0
    obs, _ = env.reset()
    
    h_t = torch.zeros(1, 256, device=device)
    z_t = torch.zeros(1, 32*32, device=device)
    prev_act_env = torch.zeros(1, action_dim, device=device)
    
    while global_step < 100000:
        with torch.no_grad():
            encoded_obs = encoder(torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0))
            h_t, _ = rssm.step_prior(h_t, z_t, prev_act_env)
            _, z_t = rssm.step_posterior(h_t, encoded_obs)
            
            action_dist = actor(h_t, z_t)
            # Add normal exploration noise or sample
            action = action_dist.sample().clamp(-2.0, 2.0).squeeze(0).cpu().numpy()
            
        next_obs, reward, done, trunc, _ = env.step(action)
        buffer.add(obs, action, reward, done)
        
        ep_return += reward
        ep_len += 1
        obs = next_obs
        prev_act_env = torch.tensor(action, dtype=torch.float32, device=device).unsqueeze(0)
        
        global_step += 1
        
        if done or trunc or ep_len >= 200:
            writer.add_scalar("Env/Return", ep_return, global_step)
            print(f"Step {global_step} | Return: {ep_return:.2f}")
            obs, _ = env.reset()
            h_t = torch.zeros(1, 256, device=device)
            z_t = torch.zeros(1, 32*32, device=device)
            prev_act_env = torch.zeros(1, action_dim, device=device)
            ep_return = 0
            ep_len = 0

        # Train
        if global_step % 1 == 0:
            batch_obs, batch_act, batch_rew, batch_done = buffer.sample_batch_sequence(batch_size=32, seq_len=50)
            batch_obs = batch_obs.to(device)
            batch_act = batch_act.to(device)
            batch_rew = batch_rew.to(device)
            
            h_train = torch.zeros(32, 256, device=device)
            z_train = torch.zeros(32, 32*32, device=device)
            
            loss_decoder, loss_reward, loss_kl = 0, 0, 0
            encoded_obs_train = encoder(batch_obs)
            
            posteriors_z = []
            histories_h = []
            
            prev_act = torch.zeros(32, action_dim, device=device)
            for t in range(50):
                h_train, prior_logits = rssm.step_prior(h_train, z_train, prev_act)
                post_logits, z_train = rssm.step_posterior(h_train, encoded_obs_train[t])
                
                posteriors_z.append(z_train)
                histories_h.append(h_train)
                
                loss_kl += kl_balancing_loss(prior_logits, post_logits)
                loss_decoder += decoder.loss(h_train, z_train, batch_obs[t])
                loss_reward += reward_pred.loss(h_train, z_train, batch_rew[t])
                
                prev_act = batch_act[t]
                
            loss_model = (loss_kl / 50) + (loss_decoder / 50) + (loss_reward / 50)
            opt_model.zero_grad()
            loss_model.backward()
            torch.nn.utils.clip_grad_norm_(model_params, 1000.0)
            opt_model.step()
            
            flat_h = torch.cat(histories_h, dim=0).detach() 
            flat_z = torch.cat(posteriors_z, dim=0).detach()
            
            imagined_h = flat_h.contiguous()
            imagined_z = flat_z.contiguous()
            
            horizon_h = []
            horizon_z = []
            horizon_rewards = []
            
            horizon_h.append(imagined_h)
            horizon_z.append(imagined_z)
            
            for _ in range(15):
                imagined_action = actor(imagined_h, imagined_z).rsample()
                imagined_h, prior_logits = rssm.step_prior(imagined_h, imagined_z, imagined_action)
                imagined_z = st_sample(prior_logits).view(-1, 32*32) 
                
                pred_reward = reward_pred.predict(imagined_h, imagined_z)
                
                horizon_h.append(imagined_h)
                horizon_z.append(imagined_z)
                horizon_rewards.append(pred_reward)
                
            stack_h = torch.stack(horizon_h)
            stack_z = torch.stack(horizon_z)
            imagined_values = critic.get_value(stack_h, stack_z).squeeze(-1) 
            stack_rew = torch.stack(horizon_rewards).squeeze(-1)
            
            target_returns = compute_lambda_returns(stack_rew, imagined_values[1:].detach(), gamma=0.99, lambda_=0.95).unsqueeze(-1)
            
            loss_actor = -target_returns.mean()
            
            for p in critic.parameters():
                p.requires_grad = False
                
            opt_actor.zero_grad()
            loss_actor.backward()
            opt_actor.step()
            
            for p in critic.parameters():
                p.requires_grad = True

            loss_critic = critic.loss(stack_h[1:-1].detach(), stack_z[1:-1].detach(), target_returns.detach())
            opt_critic.zero_grad()
            loss_critic.backward()
            opt_critic.step()

            if global_step % 500 == 0:
                print(f"Step {global_step} | Model: {loss_model.item():.2f} | Actor: {loss_actor.item():.2f} | Critic: {loss_critic.item():.2f} | KL: {(loss_kl/50).item():.2f}")
                writer.add_scalar("Loss/Model", loss_model.item(), global_step)
                writer.add_scalar("Loss/Actor", loss_actor.item(), global_step)
                writer.add_scalar("Loss/Critic", loss_critic.item(), global_step)
                writer.add_scalar("Loss/KL", (loss_kl/50).item(), global_step)

    writer.close()
    print("DreamerV3 Real Algorithm End-to-End Run Completed!")
"""

with open("06_dreamer_v3.py", "w") as f:
    f.write(prefix + new_loop)
