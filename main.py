from DQN import DQN

def main():
	breakout = DQN()
	breakout.train()
	#breakout.demo()

if __name__ == "__main__":
	main()