from DQN import DQN

def main():
	breakout = DQN()	
	breakout.train(
		discount=0.99,
		replay_memory_init_size=50000,
		epsilon_start=1.0,epsilon_end=0.1,
		epsilon_decay_steps=500000,
		monitor_record_steps=250,
		max_replay=800000,
		restore=False)
	#breakout.demo()

if __name__ == "__main__":
	main()