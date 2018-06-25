from DQN import DQN

def main():

	momentum = [0.9]

	breakout = DQN(env_name="Breakout-v0",run_id="maxqhuber1",momentum=0.95)	

	breakout.train(
		discount=0.99,
		replay_memory_init_size=50000,
		epsilon_start=1.0,
		epsilon_end=0.1,
		epsilon_decay_steps=500000,
		monitor_record_steps=50,
		max_replay=800000,
		num_episodes=15000,
		restore=False,
		start_counter=0)

	print("Done")


if __name__ == "__main__":
	main()