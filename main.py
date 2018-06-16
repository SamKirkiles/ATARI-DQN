from DQN import DQN

def main():
	breakout = DQN()
	breakout.train(restore=False)
	#breakout.demo()

if __name__ == "__main__":
	main()